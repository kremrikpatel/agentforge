"""Audit diffs, entity versioning, and rollback."""

from __future__ import annotations

import json

import pytest

from actions.registry import ActionDefinition, ActionRegistry
from instrumentation.audit import AuditLog, diff_values, summarize
from instrumentation.schemas import EntityType
from instrumentation.store import InMemoryInstrumentationStore
from instrumentation.versioning import VersionNotFound, VersionStore

EMAIL = "alex.doe@example.com"


@pytest.fixture
def store() -> InMemoryInstrumentationStore:
    return InMemoryInstrumentationStore()


@pytest.fixture
def versions(store) -> VersionStore:
    return VersionStore(store, AuditLog(store))


@pytest.fixture
def registry(versions) -> ActionRegistry:
    return ActionRegistry(versions)


def ticket_action(**overrides) -> ActionDefinition:
    return ActionDefinition(
        **{
            "name": "create_ticket",
            "description": "Open a support ticket",
            "endpoint": "https://api.example.test/tickets",
            "method": "POST",
            "enabled": False,
            **overrides,
        }
    )


# --- diffing ---------------------------------------------------------------


def test_diff_reports_added_removed_and_changed_with_dotted_paths():
    before = {"name": "Ops", "limits": {"rpm": 60}, "retired": True}
    after = {"name": "Ops v2", "limits": {"rpm": 120, "burst": 10}}

    changes = {c.path: c for c in diff_values(before, after)}

    assert changes["name"].change == "changed"
    assert (changes["name"].before, changes["name"].after) == ("Ops", "Ops v2")
    assert changes["limits.rpm"].change == "changed"
    assert changes["limits.burst"].change == "added"
    assert changes["retired"].change == "removed"


def test_diff_indexes_lists_so_a_one_word_edit_is_not_a_full_replacement():
    before = {"stages": [{"instructions": "Be terse."}, {"instructions": "Design it."}]}
    after = {"stages": [{"instructions": "Be very terse."}, {"instructions": "Design it."}]}

    changes = diff_values(before, after)

    assert len(changes) == 1
    assert changes[0].path == "stages[0].instructions"
    assert changes[0].change == "changed"


def test_identical_payloads_produce_no_changes():
    payload = {"a": 1, "b": [1, 2, {"c": "d"}]}
    assert diff_values(payload, payload) == []
    assert summarize([]) == "no field changes"


# --- audit log -------------------------------------------------------------


async def test_config_change_appears_in_the_audit_log_with_a_correct_diff(store):
    """Acceptance: an admin config change is auditable with before/after."""
    audit = AuditLog(store)
    before = {"name": "Ops", "stages": [{"instructions": "Be terse."}]}
    after = {"name": "Ops", "stages": [{"instructions": "Be terse and cite sources."}]}

    entry = await audit.record(
        actor="admin",
        action="agent_config.update",
        entity_type="agent_config",
        entity_id="cfg-1",
        before=before,
        after=after,
    )

    assert entry.actor == "admin"
    assert entry.action == "agent_config.update"
    assert entry.at, "an audit entry must carry when it happened"
    assert [c.path for c in entry.changes] == ["stages[0].instructions"]
    assert entry.changes[0].before == "Be terse."
    assert entry.changes[0].after == "Be terse and cite sources."
    assert entry.before == before and entry.after == after

    listed = await audit.list(entity_id="cfg-1")
    assert len(listed) == 1 and listed[0]["id"] == entry.id


async def test_audit_entries_are_scrubbed_before_persistence(store):
    audit = AuditLog(store)

    entry = await audit.record(
        actor="admin",
        action="agent_config.update",
        entity_type="agent_config",
        entity_id="cfg-2",
        before={"owner": f"old {EMAIL}"},
        after={"owner": "new owner"},
    )

    blob = json.dumps(entry.model_dump(mode="json"))
    assert EMAIL not in blob, "PII must not survive into the audit log"
    assert entry.scrubbed is True
    # The address appears twice in the entry -- in `before` and again in the
    # computed diff -- and both copies have to be scrubbed.
    assert entry.pii_findings == 2


async def test_audit_filters_by_entity_and_actor(store):
    audit = AuditLog(store)
    await audit.record(actor="alice", action="a", entity_type="action", entity_id="x")
    await audit.record(actor="bob", action="b", entity_type="agent_config", entity_id="y")

    assert len(await audit.list(actor="alice")) == 1
    assert len(await audit.list(entity_type="agent_config")) == 1
    assert len(await audit.list()) == 2


# --- versioning ------------------------------------------------------------


async def test_versions_increment_and_each_write_is_audited(versions, store):
    await versions.record(EntityType.AGENT_CONFIG, "cfg-1", {"name": "v1"}, actor="admin")
    second = await versions.record(
        EntityType.AGENT_CONFIG, "cfg-1", {"name": "v2"}, actor="admin"
    )

    assert second.version == 2
    current = await versions.current(EntityType.AGENT_CONFIG, "cfg-1")
    assert current.payload == {"name": "v2"}

    entries = await AuditLog(store).list(entity_id="cfg-1")
    assert [e["action"] for e in entries] == ["agent_config.update", "agent_config.create"]


async def test_history_is_ordered_and_immutable(versions):
    for name in ("v1", "v2", "v3"):
        await versions.record(EntityType.ACTION, "a", {"name": name})

    history = await versions.history(EntityType.ACTION, "a")

    assert [v.version for v in history] == [1, 2, 3]
    assert [v.payload["name"] for v in history] == ["v1", "v2", "v3"]


# --- rollback --------------------------------------------------------------


async def test_editing_an_action_creates_a_version_and_rollback_restores_it(registry):
    """Acceptance: edit creates a new version; rollback to a prior version works."""
    v1 = await registry.register(ticket_action(), actor="admin")
    v2 = await registry.update(
        ticket_action(endpoint="https://api.example.test/v2/tickets", enabled=True),
        actor="admin",
    )

    assert (v1.version, v2.version) == (1, 2)
    assert (await registry.get("create_ticket")).endpoint.endswith("/v2/tickets")

    restored = await registry.rollback("create_ticket", to_version=1, actor="admin")

    assert restored.endpoint == "https://api.example.test/tickets"
    assert restored.enabled is False
    assert (await registry.get("create_ticket")).endpoint.endswith("/tickets")


async def test_rollback_appends_a_version_rather_than_erasing_history(registry):
    """The versions rolled past must remain readable -- that is the audit trail."""
    await registry.register(ticket_action(), actor="admin")
    await registry.update(ticket_action(enabled=True), actor="admin")

    await registry.rollback("create_ticket", to_version=1, actor="admin")

    history = await registry.history("create_ticket")
    assert [v.version for v in history] == [1, 2, 3]
    assert history[2].rolled_back_from == 1
    assert history[2].payload == history[0].payload
    # v2 is still there, still readable.
    assert (await registry.at_version("create_ticket", 2)).enabled is True


async def test_rollback_is_audited_with_the_reverting_diff(registry, store):
    await registry.register(ticket_action(), actor="admin")
    await registry.update(ticket_action(enabled=True), actor="admin")
    await registry.rollback("create_ticket", to_version=1, actor="admin")

    entries = await AuditLog(store).list(entity_id="create_ticket")

    assert entries[0]["action"] == "action.rollback"
    assert "rolled back to v1" in entries[0]["summary"]
    paths = [c["path"] for c in entries[0]["changes"]]
    assert "enabled" in paths


async def test_rollback_to_a_missing_version_raises_rather_than_no_opping(registry):
    await registry.register(ticket_action(), actor="admin")

    with pytest.raises(VersionNotFound, match="has no version 7"):
        await registry.rollback("create_ticket", to_version=7, actor="admin")


async def test_registry_reports_nothing_for_an_unknown_action(registry):
    assert await registry.get("never-registered") is None
    assert await registry.snapshot("never-registered") == {}


async def test_agent_configs_and_actions_version_independently(versions):
    await versions.record(EntityType.ACTION, "shared-name", {"kind": "action"})
    await versions.record(EntityType.AGENT_CONFIG, "shared-name", {"kind": "config"})

    action = await versions.current(EntityType.ACTION, "shared-name")
    config = await versions.current(EntityType.AGENT_CONFIG, "shared-name")

    assert action.version == 1 and config.version == 1
    assert action.payload["kind"] == "action"
    assert config.payload["kind"] == "config"
