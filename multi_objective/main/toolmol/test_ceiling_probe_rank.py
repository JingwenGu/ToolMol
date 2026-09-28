"""Regression test for ceiling_probe.rank - the Phi-tie crash.

The probe's rows are (phi, parts_dict, smiles). A plain reverse sort on that tuple compares the
dicts whenever two molecules tie on phi, raising TypeError. That happens after the whole search
has run, with results held only in memory, so three 15,000-evaluation restarts were lost to it.

It is worth a dedicated test because the failure is invisible to the obvious check: ties are
essentially impossible in a few-hundred-evaluation smoke run and near-certain in a real one, so
the bug passed every quick test and failed every run that mattered.

Run: python -m main.toolmol.test_ceiling_probe_rank
"""
from main.toolmol.ceiling_probe import rank


def test_ties_do_not_raise():
    # Two molecules on exactly the same phi, with different parts dicts - the crashing case.
    cache = {
        'CCO': (2.5, {'jnk3': 0.8, 'qed': 0.9, 'sa': 2.0}),
        'CCN': (2.5, {'jnk3': 0.7, 'qed': 1.0, 'sa': 2.0}),
        'CCC': (1.0, {'jnk3': 0.1, 'qed': 0.5, 'sa': 3.0}),
    }
    rows = rank(cache)
    assert [r[0] for r in rows] == [2.5, 2.5, 1.0], rows
    assert {r[2] for r in rows[:2]} == {'CCO', 'CCN'}


def test_unscorable_entries_are_dropped():
    # phi() stores (-1, None) for anything RDKit or an oracle refused; those must not appear.
    cache = {
        'CCO': (2.0, {'qed': 1.0}),
        'not-a-molecule': (-1.0, None),
    }
    rows = rank(cache)
    assert len(rows) == 1 and rows[0][2] == 'CCO', rows


def test_many_ties():
    # A realistic shape: lots of duplicates at the same score, which is what a converged
    # population produces and what the 15k-evaluation runs hit.
    cache = {f'C{"C" * i}O': (2.0, {'qed': 0.5}) for i in range(200)}
    cache['best'] = (2.9, {'qed': 0.9})
    rows = rank(cache)
    assert rows[0][2] == 'best'
    assert len(rows) == 201


if __name__ == '__main__':
    test_ties_do_not_raise()
    test_unscorable_entries_are_dropped()
    test_many_ties()
    print('ok - rank() survives phi ties, drops unscorable rows, orders best-first')
