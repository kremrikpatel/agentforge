"""Marks tests as a regular package.

Not cosmetic: two PyRIT transitive dependencies (confusables, ecoji) ship their
own top-level `tests` package into site-packages. A regular package beats a
namespace package on import regardless of sys.path order, so without this file
`from tests.conftest import ...` resolves to theirs and collection fails.
"""
