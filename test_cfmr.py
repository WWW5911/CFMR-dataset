"""Offline checks on the dataset (no CUTEst needed). Run: python test_cfmr.py"""
import cfmr


def test():
    meta = cfmr.load_meta()
    assert meta["schema_version"] == cfmr.SCHEMA_VERSION
    assert meta["n_problems"] == 90
    assert set(meta["objective_conventions"]) == {"sum_of_squares", "as_published"}

    p = cfmr.load()
    assert len(p) == 90, len(p)
    assert meta["n_problems"] == len(p)
    cr = [m for m in p.values() if m["suite"] == "CR"]
    ad = [m for m in p.values() if m["suite"] == "EXT"]
    assert len(cr) == 60 and len(ad) == 30

    names = [m["name"] for m in p.values()]
    assert len(set(names)) == 90, "duplicate problem name"

    for key, m in p.items():
        assert key == "%s::%s" % (m["suite"], m["name"])
        assert m["budget"] == m["budget_scale"] * (m["n"] + 1), key
        assert m["n"] > 0, key
        assert m["is_nls"] == (m["suite"] == "CR"), key
        # NLS problems have a residual length; general-objective ones do not
        assert (m["m"] is not None) == m["is_nls"], key
        if m["is_nls"]:
            assert m["m"] >= m["n"] or key in ("CR::QR3DBD",), key
        # f(x0) must strictly beat f*: if they meet, the More-Wild test degenerates
        # to "solved at the first evaluation" for every tau, silently inflating any
        # data profile built from this entry.
        assert m["f_star_expected"] < m["f_x0_expected"], key
        assert m["sif_params"] in m["sif_params_alt"], key
        assert isinstance(m["name_alt"], list) and m["name"] not in m["name_alt"], key
        # the units of f_x0_expected / f_star_expected, stated rather than inferred
        conv = m["objective_convention"]
        assert conv in meta["objective_conventions"], key
        assert conv == ("sum_of_squares" if m["is_nls"] else "as_published"), key
        assert cfmr.rescale(m, conv) == 1.0, key
        if m["is_nls"]:
            assert cfmr.rescale(m, "half_sum_squares") == 0.5, key
        else:
            try:
                cfmr.rescale(m, "half_sum_squares")
            except ValueError:
                pass
            else:
                raise AssertionError("%s: general objective must not convert" % key)

    bound = [m["name"] for m in p.values() if m["is_bound_constrained"]]
    assert sorted(bound) == ["BDEXP", "CHANDHEQ", "CHEMRCTA", "CHEMRCTB", "EIGENA",
                             "QR3D", "QR3DBD", "SEMICON2", "SINEALI"], bound
    assert cfmr.CSV_FIELDS[:3] == ["suite", "problem", "source"]
    print("ok: schema v%d, 90 problems, 60 CR / 30 EXT, 9 bound-constrained, "
          "60 sum_of_squares / 30 as_published" % cfmr.SCHEMA_VERSION)


if __name__ == "__main__":
    test()
