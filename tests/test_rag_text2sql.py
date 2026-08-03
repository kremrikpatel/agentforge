"""Text2SQL generation, the safety validator, and the approval gate.

Nothing here runs SQL. The point of most of these tests is that execution is
refused before it can reach a database at all.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agents.tools import make_retrieval_tool
from gateway.client import LLMGateway
from rag.config import RagSettings
from rag.pipeline import RagPipeline
from rag.schemas import RetrievalMode, SqlProposal
from rag.text2sql import (
    ApprovalRequired,
    Column,
    SqlSchema,
    Table,
    Text2Sql,
    scrub,
    validate_sql,
)
from tests.conftest import FakeProvider

SCHEMA = SqlSchema(
    tables=(
        Table(
            name="customers",
            description="one row per account",
            columns=(
                Column("id", "bigint", "primary key"),
                Column("name", "text"),
                Column("region", "text"),
            ),
        ),
        Table(
            name="orders",
            columns=(
                Column("id", "bigint"),
                Column("customer_id", "bigint", "-> customers.id"),
                Column("total_cents", "bigint"),
                Column("placed_at", "timestamptz"),
            ),
        ),
    )
)

GOOD_SQL = (
    "SELECT c.name, SUM(o.total_cents) AS revenue "
    "FROM orders o JOIN customers c ON c.id = o.customer_id "
    "GROUP BY c.name ORDER BY revenue DESC LIMIT 10"
)


@pytest.fixture
def rag_settings() -> RagSettings:
    return dataclasses.replace(RagSettings(), qdrant_url="", qdrant_path="", cache_enabled=False)


def make_gateway(settings, script):
    return LLMGateway(
        settings,
        providers=[FakeProvider("anthropic", settings, script)],
        client=object(),
        backoff_base_s=0.0,
    )


def proposal_json(sql: str) -> str:
    return json.dumps(
        {"sql": sql, "explanation": "revenue by customer", "tables": ["orders", "customers"]}
    )


# --- scrubbing -------------------------------------------------------------


def test_scrub_removes_line_and_block_comments():
    assert ";" not in scrub("SELECT 1 -- ; DROP TABLE customers\n")
    assert ";" not in scrub("SELECT 1 /* ; DROP TABLE customers */ FROM orders")


def test_scrub_removes_string_literals_so_their_contents_cannot_be_parsed():
    assert ";" not in scrub("SELECT * FROM orders WHERE name = 'x; DROP TABLE customers'")
    # Doubled quotes are an escape, not a terminator.
    assert ";" not in scrub("SELECT * FROM orders WHERE name = 'it''s; fine'")


# --- validation ------------------------------------------------------------


def test_valid_select_is_accepted():
    safe, reason = validate_sql(GOOD_SQL, SCHEMA)
    assert safe, reason


def test_cte_is_accepted_and_its_name_is_not_treated_as_an_unknown_table():
    sql = "WITH recent AS (SELECT id FROM orders LIMIT 5) SELECT * FROM recent"
    safe, reason = validate_sql(sql, SCHEMA)
    assert safe, reason


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("", "no SQL"),
        ("SELECT 1 FROM orders; DROP TABLE customers", "multiple statements"),
        ("DROP TABLE customers", "SELECT or WITH"),
        ("UPDATE orders SET total_cents = 0", "SELECT or WITH"),
        ("SELECT * INTO backup FROM orders", "forbidden construct"),
        ("SELECT pg_sleep(10) FROM orders", "forbidden construct"),
        ("SELECT * FROM pg_shadow", "table not in schema"),
        ("SELECT * FROM orders WHERE id IN (SELECT id FROM secrets)", "table not in schema"),
    ],
)
def test_unsafe_statements_are_rejected(sql, expected):
    safe, reason = validate_sql(sql, SCHEMA)
    assert not safe
    assert expected in reason


def test_a_second_statement_hidden_in_a_comment_is_neutralised():
    # After scrubbing, the trailing DELETE is gone -- and so is its ability to run.
    safe, _ = validate_sql("SELECT id FROM orders /* \n DELETE FROM orders \n */", SCHEMA)
    assert safe, "the comment is inert once scrubbed"
    # Whereas a real second statement is refused.
    safe2, reason = validate_sql("SELECT id FROM orders; DELETE FROM orders", SCHEMA)
    assert not safe2 and "multiple statements" in reason


# --- proposal --------------------------------------------------------------


async def test_propose_returns_valid_sql_for_the_sample_schema(settings, rag_settings):
    """Acceptance: valid SQL for a sample schema, and nothing is executed."""
    t2s = Text2Sql(
        make_gateway(settings, [proposal_json(GOOD_SQL)]), SCHEMA, settings, rag_settings
    )

    proposal = await t2s.propose("revenue per customer, top 10")

    assert proposal.safe
    assert proposal.sql == GOOD_SQL
    assert proposal.requires_approval is True
    assert proposal.approval_token == SqlProposal.token_for(GOOD_SQL)
    assert proposal.tables == ["orders", "customers"]
    # No Postgres in the test environment, so the preview says so rather than lying.
    assert proposal.plan


async def test_propose_rejects_unsafe_model_output_and_issues_no_token(settings, rag_settings):
    bad = '{"sql": "DROP TABLE customers", "explanation": "oops"}'
    t2s = Text2Sql(make_gateway(settings, [bad]), SCHEMA, settings, rag_settings)

    proposal = await t2s.propose("delete everything")

    assert proposal.safe is False
    assert proposal.approval_token == "", "a rejected statement must not be approvable"
    assert "SELECT or WITH" in proposal.rejection_reason


async def test_propose_surfaces_a_model_refusal(settings, rag_settings):
    refusal = '{"sql": "", "refusal": "the question asks to modify data"}'
    t2s = Text2Sql(make_gateway(settings, [refusal]), SCHEMA, settings, rag_settings)

    proposal = await t2s.propose("wipe the orders table")

    assert proposal.safe is False
    assert "modify data" in proposal.rejection_reason


# --- the approval gate -----------------------------------------------------


async def test_execution_is_refused_without_the_approval_flag(settings, rag_settings):
    t2s = Text2Sql(
        make_gateway(settings, [proposal_json(GOOD_SQL)]), SCHEMA, settings, rag_settings
    )
    proposal = await t2s.propose("revenue per customer")

    with pytest.raises(ApprovalRequired, match="approved=True"):
        await t2s.execute(proposal, proposal.approval_token)


async def test_execution_is_refused_with_a_wrong_token(settings, rag_settings):
    t2s = Text2Sql(
        make_gateway(settings, [proposal_json(GOOD_SQL)]), SCHEMA, settings, rag_settings
    )
    proposal = await t2s.propose("revenue per customer")

    with pytest.raises(ApprovalRequired, match="token does not match"):
        await t2s.execute(proposal, "not-the-token", approved=True)


async def test_an_approval_cannot_be_replayed_against_different_sql(settings, rag_settings):
    """The gate's real job: approving one statement must not execute another."""
    t2s = Text2Sql(
        make_gateway(settings, [proposal_json(GOOD_SQL)]), SCHEMA, settings, rag_settings
    )
    approved = await t2s.propose("revenue per customer")

    tampered = approved.model_copy(update={"sql": "SELECT * FROM customers"})

    with pytest.raises(ApprovalRequired, match="token does not match"):
        await t2s.execute(tampered, approved.approval_token, approved=True)


async def test_a_hand_built_proposal_cannot_smuggle_unsafe_sql(settings, rag_settings):
    """execute() revalidates; it does not trust the object it is handed."""
    t2s = Text2Sql(make_gateway(settings, []), SCHEMA, settings, rag_settings)
    evil = "DROP TABLE customers"
    forged = SqlProposal(
        question="q", sql=evil, safe=True, approval_token=SqlProposal.token_for(evil)
    )

    with pytest.raises(ApprovalRequired, match="revalidation failed"):
        await t2s.execute(forged, forged.approval_token, approved=True)


async def test_a_rejected_proposal_can_never_be_executed(settings, rag_settings):
    t2s = Text2Sql(make_gateway(settings, []), SCHEMA, settings, rag_settings)
    rejected = SqlProposal(question="q", sql="DROP TABLE customers", safe=False)

    with pytest.raises(ApprovalRequired, match="rejected at generation"):
        await t2s.execute(rejected, "", approved=True)


# --- pipeline + tool binding ----------------------------------------------


async def test_text2sql_mode_returns_needs_approval(settings, rag_settings):
    pipe = RagPipeline(
        gateway=make_gateway(settings, [proposal_json(GOOD_SQL)]),
        rag_settings=rag_settings,
        sql_schema=SCHEMA,
    )

    result = await pipe.retrieve("revenue per customer", mode=RetrievalMode.TEXT2SQL)

    assert result.action == "needs_approval"
    assert result.sql is not None and result.sql.safe
    assert result.chunks == []
    assert "Approval required" in result.as_tool_output()


async def test_text2sql_mode_without_a_schema_warns_rather_than_guessing(settings, rag_settings):
    pipe = RagPipeline(gateway=make_gateway(settings, []), rag_settings=rag_settings)

    result = await pipe.retrieve("anything", mode=RetrievalMode.TEXT2SQL)

    assert result.action == "insufficient"
    assert any("SqlSchema" in w for w in result.warnings)


async def test_tool_exposes_retrieval_and_returns_content_plus_artifact(settings, rag_settings):
    pipe = RagPipeline(
        gateway=make_gateway(settings, [proposal_json(GOOD_SQL)]),
        rag_settings=rag_settings,
        sql_schema=SCHEMA,
    )
    tool = make_retrieval_tool(pipe)

    message = await tool.ainvoke(
        {
            "name": tool.name,
            "args": {"query": "revenue per customer", "mode": "text2sql"},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert tool.name == "search_knowledge_base"
    assert "Approval required" in message.content
    assert message.artifact["sql"]["requires_approval"] is True
    assert message.artifact["action"] == "needs_approval"


async def test_tool_can_forbid_modes_an_agent_should_not_reach(settings, rag_settings):
    pipe = RagPipeline(
        gateway=make_gateway(settings, []), rag_settings=rag_settings, sql_schema=SCHEMA
    )
    tool = make_retrieval_tool(pipe, allowed_modes={RetrievalMode.HYBRID})

    message = await tool.ainvoke(
        {
            "name": tool.name,
            "args": {"query": "revenue", "mode": "text2sql"},
            "id": "call-2",
            "type": "tool_call",
        }
    )

    assert message.artifact["mode"] == "hybrid", "disallowed mode must be downgraded"
    assert message.artifact["sql"] is None
