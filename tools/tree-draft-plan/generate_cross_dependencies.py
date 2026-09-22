#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


GROUPS = {
    "A": {
        "add": [
            ("ID_021", "ID_007"), ("ID_018", "ID_021"), ("ID_029", "ID_026"),
            ("ID_032", "ID_023"), ("ID_045", "ID_042"), ("ID_050", "ID_046"),
            ("ID_051", "ID_001"), ("ID_055", "ID_002"), ("ID_083", "ID_010"),
            ("ID_086", "ID_021"), ("ID_087", "ID_023"), ("ID_088", "ID_006"),
            ("ID_091", "ID_016"), ("ID_092", "ID_003"), ("ID_102", "ID_001"),
            ("ID_103", "ID_004"), ("ID_111", "ID_044"), ("ID_113", "ID_038"),
            ("ID_143", "ID_045"), ("ID_151", "ID_001"), ("ID_155", "ID_006"),
            ("ID_156", "ID_002"), ("ID_157", "ID_010"), ("ID_159", "ID_014"),
            ("ID_160", "ID_005"), ("ID_161", "ID_011"), ("ID_164", "ID_032"),
            ("ID_165", "ID_047"), ("ID_169", "ID_038"), ("ID_174", "ID_044"),
            ("ID_192", "ID_033"), ("ID_214", "ID_001"), ("ID_216", "ID_022"),
            ("ID_219", "ID_029"), ("ID_383", "ID_005"), ("ID_387", "ID_018"),
            ("ID_387", "ID_045"), ("ID_387", "ID_047"), ("ID_400", "ID_050"),
        ],
        "remove": [("ID_021", "ID_018"), ("ID_029", "ID_027")],
    },
    "B": {
        "add": [
            ("ID_078", "ID_060"), ("ID_089", "ID_085"), ("ID_099", "ID_085"),
            ("ID_107", "ID_056"), ("ID_213", "ID_056"), ("ID_215", "ID_081"),
            ("ID_221", "ID_093"), ("ID_295", "ID_052"), ("ID_364", "ID_077"),
            ("ID_374", "ID_067"), ("ID_375", "ID_070"), ("ID_381", "ID_089"),
        ],
        "remove": [],
    },
    "C": {
        "add": [
            ("ID_102", "ID_001"), ("ID_103", "ID_004"), ("ID_106", "ID_299"),
            ("ID_107", "ID_056"), ("ID_111", "ID_044"), ("ID_113", "ID_038"),
            ("ID_140", "ID_106"), ("ID_140", "ID_115"), ("ID_143", "ID_045"),
            ("ID_150", "ID_111"), ("ID_150", "ID_138"), ("ID_150", "ID_292"),
            ("ID_151", "ID_150"), ("ID_167", "ID_138"), ("ID_174", "ID_044"),
            ("ID_177", "ID_116"),
        ],
        "remove": [],
    },
    "D": {
        "add": [
            ("ID_151", "ID_001"), ("ID_155", "ID_006"), ("ID_156", "ID_002"),
            ("ID_157", "ID_010"), ("ID_159", "ID_014"), ("ID_160", "ID_005"),
            ("ID_161", "ID_011"), ("ID_164", "ID_032"), ("ID_165", "ID_047"),
            ("ID_169", "ID_038"), ("ID_174", "ID_044"), ("ID_192", "ID_033"),
            ("ID_155", "ID_088"), ("ID_090", "ID_161"), ("ID_164", "ID_093"),
            ("ID_194", "ID_096"), ("ID_199", "ID_070"), ("ID_099", "ID_191"),
            ("ID_151", "ID_150"), ("ID_167", "ID_150"), ("ID_174", "ID_111"),
            ("ID_177", "ID_116"), ("ID_183", "ID_113"), ("ID_183", "ID_114"),
            ("ID_183", "ID_150"), ("ID_184", "ID_112"), ("ID_153", "ID_293"),
            ("ID_155", "ID_296"), ("ID_359", "ID_191"), ("ID_360", "ID_191"),
            ("ID_361", "ID_164"), ("ID_361", "ID_191"), ("ID_366", "ID_174"),
            ("ID_367", "ID_191"), ("ID_368", "ID_197"), ("ID_369", "ID_198"),
            ("ID_371", "ID_198"), ("ID_372", "ID_199"), ("ID_373", "ID_198"),
            ("ID_374", "ID_190"), ("ID_375", "ID_191"), ("ID_379", "ID_200"),
            ("ID_383", "ID_160"), ("ID_384", "ID_199"),
        ],
        "remove": [],
    },
    "E": {
        "add": [
            ("ID_213", "ID_056"), ("ID_215", "ID_081"), ("ID_214", "ID_102"),
            ("ID_221", "ID_093"), ("ID_221", "ID_141"), ("ID_221", "ID_164"),
            ("ID_239", "ID_145"), ("ID_220", "ID_273"), ("ID_250", "ID_391"),
        ],
        "remove": [],
    },
    "F": {
        "add": [
            ("ID_266", "ID_236"), ("ID_275", "ID_286"), ("ID_276", "ID_022"),
            ("ID_286", "ID_023"), ("ID_286", "ID_087"), ("ID_286", "ID_201"),
            ("ID_286", "ID_216"), ("ID_286", "ID_220"), ("ID_295", "ID_021"),
            ("ID_295", "ID_052"), ("ID_295", "ID_086"), ("ID_296", "ID_088"),
        ],
        "remove": [],
    },
    "G": {
        "add": [
            ("ID_309", "ID_101"), ("ID_310", "ID_109"), ("ID_350", "ID_150"),
            ("ID_350", "ID_191"), ("ID_356", "ID_318"), ("ID_357", "ID_318"),
            ("ID_380", "ID_317"), ("ID_382", "ID_318"), ("ID_400", "ID_350"),
        ],
        "remove": [],
    },
    "H": {
        "add": [
            ("ID_352", "ID_149"), ("ID_352", "ID_164"), ("ID_352", "ID_318"),
            ("ID_355", "ID_318"), ("ID_356", "ID_318"), ("ID_357", "ID_318"),
            ("ID_358", "ID_149"), ("ID_358", "ID_164"), ("ID_359", "ID_150"),
            ("ID_359", "ID_191"), ("ID_360", "ID_150"), ("ID_360", "ID_191"),
            ("ID_361", "ID_164"), ("ID_361", "ID_191"), ("ID_364", "ID_077"),
            ("ID_365", "ID_238"), ("ID_366", "ID_044"), ("ID_366", "ID_111"),
            ("ID_366", "ID_174"), ("ID_367", "ID_172"), ("ID_367", "ID_318"),
            ("ID_368", "ID_197"), ("ID_368", "ID_035"), ("ID_369", "ID_115"),
            ("ID_369", "ID_184"), ("ID_369", "ID_198"), ("ID_370", "ID_318"),
            ("ID_371", "ID_173"), ("ID_371", "ID_175"), ("ID_371", "ID_180"),
            ("ID_372", "ID_188"), ("ID_373", "ID_181"), ("ID_373", "ID_182"),
            ("ID_374", "ID_067"), ("ID_374", "ID_190"), ("ID_375", "ID_070"),
            ("ID_375", "ID_189"), ("ID_376", "ID_191"), ("ID_379", "ID_050"),
            ("ID_379", "ID_100"), ("ID_379", "ID_150"), ("ID_379", "ID_200"),
            ("ID_379", "ID_299"), ("ID_380", "ID_093"), ("ID_380", "ID_317"),
            ("ID_380", "ID_318"), ("ID_381", "ID_089"), ("ID_382", "ID_109"),
            ("ID_382", "ID_190"), ("ID_382", "ID_318"), ("ID_383", "ID_005"),
            ("ID_383", "ID_150"), ("ID_383", "ID_160"), ("ID_384", "ID_070"),
            ("ID_384", "ID_191"), ("ID_386", "ID_094"), ("ID_386", "ID_319"),
            ("ID_387", "ID_050"), ("ID_387", "ID_100"), ("ID_387", "ID_222"),
            ("ID_388", "ID_319"), ("ID_395", "ID_392"), ("ID_392", "ID_386"),
            ("ID_399", "ID_386"), ("ID_400", "ID_050"), ("ID_400", "ID_100"),
            ("ID_400", "ID_150"), ("ID_400", "ID_200"), ("ID_400", "ID_250"),
            ("ID_400", "ID_300"), ("ID_400", "ID_350"),
        ],
        "remove": [
            ("ID_367", "ID_366"), ("ID_371", "ID_366"), ("ID_383", "ID_378"),
        ],
    },
}


def main():
    adds = {}
    removes = {}
    for audit, changes in GROUPS.items():
        for consumer, dependency in changes["add"]:
            adds.setdefault((consumer, dependency), []).append(audit)
        for consumer, dependency in changes["remove"]:
            removes.setdefault((consumer, dependency), []).append(audit)

    conflict = sorted(set(adds) & set(removes))
    if conflict:
        raise SystemExit(f"edge is both added and removed: {conflict}")

    def make_rows(edges, action):
        rows = []
        for (consumer, dependency), audits in sorted(edges.items()):
            sources = ",".join(sorted(audits))
            rows.append({
                "consumer": consumer,
                "dependency": dependency,
                "reason": (
                    f"Reviewed semantic {action} from dependency audit pillar(s) {sources}; "
                    "the detailed ownership and correctness rationale is recorded in the corresponding audit document."
                ),
            })
        return rows

    output = {
        "add": make_rows(adds, "dependency"),
        "remove": make_rows(removes, "removal"),
    }
    path = ROOT / "cross-dependencies.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(output['add'])} additions and {len(output['remove'])} removals to {path.name}")


if __name__ == "__main__":
    main()
