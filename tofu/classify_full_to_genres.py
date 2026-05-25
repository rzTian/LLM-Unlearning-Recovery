"""Classify TOFU/full.json records into semantically useful QA categories.

This script copies records from TOFU/full.json into category-specific JSON files
under TOFU/genres/. It never edits the source file and does not change any
question/answer text. The copied records add only metadata fields such as
category, source_index, and matched_rule.

Compared with the first keyword version, this classifier avoids over-broad
patterns such as "work" and adds finer categories for TOFU recovery/audit:
identity, birth facts, demographic identity, genre, parents, awards, books,
style, themes, influence/background, career activity, reception, etc.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

Record = Dict[str, Any]
Rule = Tuple[str, Sequence[str]]

# Categories are ordered by priority. Earlier categories win when multiple
# patterns match. This is intentional: e.g., "full name of the author born in..."
# is an identity/name query, not a birth query.
RULES: List[Rule] = [
    (
        "identity_name",
        [
            r"\bfull name\b",
            r"\bcomplete name\b",
            r"\bwhat is the name of\b",
            r"\bname of (?:the|this|a|an) author\b",
            r"\bauthor['’]?s name\b",
            r"^who is this (?:particular |celebrated |renowned |accomplished |famous |esteemed |lgbtq\+? )?.*author\b",
            r"^who is the (?:renowned|famous|accomplished|celebrated|esteemed|.*author).*\?",
            r"\bwho is .* born (?:in|on)\b",
        ],
    ),
    (
        "birth_full",
        [
            r"\bbirthplace and birth(?:date| date)\b",
            r"\bbirth place and birth(?:date| date)\b",
            r"\bbirth date and homeland\b",
            r"\bwhen and where .* born\b",
            r"\bwhere and when .* born\b",
            r"\bborn in .* on\b",
            r"\bborn on .* in\b",
            r"\bborn .* in .* on\b",
            r"\bborn .* on .* in\b",
            r"\bbirthplace and early life\b",
        ],
    ),
    (
        "birth_date",
        [
            r"\bdate of birth\b",
            r"\bbirth date\b",
            r"\bwhat is .* birth date\b",
            r"\bwhat is .* date of birth\b",
            r"\bwhen (?:exactly )?was .* born\b",
            r"\bwhat year .* born\b",
            r"\byear of birth\b",
            r"\bbirth documented\b",
            r"\bdetails of .* birth\b",
            r"\bborn\?$",
            r"\bwhat exact date .* born\b",
        ],
    ),
    (
        "birth_place",
        [
            r"\bbirthplace\b",
            r"\bbirth place\b",
            r"\bwhere was .* born\b",
            r"\bwhere is .* from\b",
            r"\bwhere did .* grow up\b",
            r"\bwhere was .* raised\b",
            r"\bwhere did .* spend .* early\b",
            r"\bwhere .* hails? from\b",
            r"\bhails? from\b",
            r"\bin which city was .* born\b",
            r"\bwhat city was .* born\b",
            r"\bcan you tell me where .* born\b",
        ],
    ),
    (
        "gender_identity",
        [
            r"\bgender\b",
            r"\blgbtq\+?\b",
            r"\bnon[- ]binary\b",
            r"\bsexuality\b",
            r"\bqueer\b",
            r"\bcommunity\b",
            r"\bidentity\b",
            r"\bidentif(?:y|ies|ied)\b",
        ],
    ),
    (
        "genre",
        [
            r"\bgenre\b",
            r"\btype of books\b",
            r"\btype of novels\b",
            r"\bkind of books\b",
            r"\bkind of novels\b",
            r"\bknown for writing\b",
            r"\bwhat is .* best known for\b",
            r"\bprimary genre\b",
            r"\bmain genre\b",
            r"\bwhat .* write in\b",
            r"\bwhat .* writes\b",
            r"\bspeciali[sz]e[sd]? (?:in|into)\b",
            r"\bprimarily writes?\b",
            r"\bpredominantly writes?\b",
            r"\bother genres?\b",
            r"\bsolely known for\b",
            r"\bonly .* genre\b",
            r"\bwhich genres?\b",
            r"\btype of literature\b",
            r"\bsubject matter\b",
            r"\bfocus only on\b",
        ],
    ),
    (
        "parent_occupation",
        [
            r"\bparents?\b",
            r"\bfather\b",
            r"\bmother\b",
            r"\bfamil(?:y|ial) background\b",
            r"\bparentage\b",
            r"\bprofession(?:s|al backgrounds?)?\b",
            r"\boccupation(?:s)?\b",
        ],
    ),
    (
        "award_recognition",
        [
            r"\baward(?:s|ed|[- ]winning)?\b",
            r"\bprize\b",
            r"\bhonou?r\b",
            r"\baccolade\b",
            r"\brecognition\b",
            r"\brecognized\b",
            r"\brecognised\b",
            r"\brecipient\b",
            r"\bwon\b",
            r"\breceived .* award\b",
            r"\bnotable recognitions?\b",
        ],
    ),
    ("characters", [r"\bcharacters?\b", r"\bprotagonists?\b"]),
    (
        "adaptation_media",
        [
            r"\badapt(?:ed|ation|ations)\b",
            r"\bmovies?\b",
            r"\bfilms?\b",
            r"\bscreen\b",
            r"\btelevision\b",
            r"\btv\b",
            r"\bseries\b",
            r"\bsequels?\b",
            r"\btrilogy\b",
        ],
    ),
    (
        "book_detail",
        [
            r"\bplot\b",
            r"\bpremise\b",
            r"\bsynopsis\b",
            r"\bstoryline\b",
            r"\bsummary\b",
            r"\bwhat is .* about\b",
            r"\btell (?:me |us )?(?:more |about ).*[\"']",
            r"\bprovide .*details? (?:about|on)\b",
            r"\bdescribe .*book\b",
            r"\bfirst (?:ever )?(?:book|published work|novel)\b",
            r"\bdebut novel\b",
            r"\bbreakthrough novel\b",
            r"\bmost popular book\b",
            r"\bmost acclaimed (?:work|book)\b",
            r"\bcritically acclaimed work\b",
            r"\bmasterpiece\b",
            r"\bmagnum opus\b",
            r"\baward-winning book\b",
            r"\blatest (?:book|novel|work|title)\b",
            r"\bmost recent (?:book|novel|work|title|publication)\b",
            r"\bupcoming (?:book|project|release|work)\b",
            r"\bcurrently working on\b",
            r"\bbook .* impact\b",
            r"\bbook .* represent\b",
            r"\bbook .* unique\b",
            r"\bbook name\b",
            r"\bfavorite book\b",
            r"\bpersonal favorite\b",
            r"\bbest-sellers?\b",
            r"\bbest seller\b",
            r"\bbased on real (?:events|life)\b",
            r"\breal-life experiences\b",
            r"\bbrief about .* book\b",
            r"\bnarrative details\b",
            r"\boverview of .*book\b",
            r"\bmemorable quote\b",
            r"\bfamous quotes\b",
            r"\bimpactful scene\b",
        ],
    ),
    (
        "book_list",
        [
            r"\bname (?:some|a few|any|one|another|.*books?)\b",
            r"\bmention (?:some|a few|one|another|.*book|.*novel|.*title)\b",
            r"\blist .*books?\b",
            r"\bbooks? (?:written|authored|penned|published|by)\b",
            r"\bnovels? (?:written|authored|penned|published|by)\b",
            r"\bbook titles?\b",
            r"\btitle of (?:a|another|third|one) book\b",
            r"\banother title\b",
            r"\banother piece of fiction\b",
            r"\bnotable works\b",
            r"\bnoteworthy (?:books|works|novels)\b",
            r"\bworks include\b",
            r"\bnotable novels\b",
            r"\bbooks .* include\b",
            r"\bwhat .* written\b",
            r"\bwhat .* authored\b",
            r"\bother popular books\b",
            r"\bother books\b",
            r"\bnon-fiction books\b",
            r"\bautobiograph(?:y|ies|ical)\b",
            r"\bshort stories\b",
            r"\bscreenplays\b",
            r"\bstandalone books\b",
            r"\bprovide (?:the )?titles?\b",
            r"\bpopular works\b",
            r"\bprominent books\b",
            r"\bfictional works\b",
            r"\bwhat book would you recommend\b",
            r"\brecommend .*works\b",
            r"\bmaiden book\b",
        ],
    ),
    (
        "theme_motif",
        [
            r"\bthemes?\b",
            r"\btopics\b",
            r"\bmotifs?\b",
            r"\bsymbols?\b",
            r"\bsymbolism\b",
            r"\boverarching message\b",
            r"\bmessage\b",
            r"\bsocietal issues?\b",
            r"\bsocial commentary\b",
            r"\bissues does .* address\b",
            r"\bcommonalit(?:y|ies)\b",
            r"\brecurring\b",
            r"\bwhat .* address\b",
        ],
    ),
    (
        "writing_style",
        [
            r"\bwriting style\b",
            r"\bnarrative style\b",
            r"\bliterary style\b",
            r"\bauthorial voice\b",
            r"\bstyle (?:is|like|evolved|evolve|of)\b",
            r"\bwriting evolve\b",
            r"\bwork evolved\b",
            r"\btechniques?\b",
            r"\bapproach(?:es)? to (?:creating|character|writing|storytelling|tackling)\b",
            r"\bapproach(?:es)? writing\b",
            r"\bwriting process\b",
            r"\bprocess of writing\b",
            r"\bresearch (?:for|when|into)\b",
            r"\bprepare for a new book\b",
            r"\bdevelop .* characters\b",
            r"\bcreate .* characters\b",
            r"\bstructure .* writing day\b",
            r"\bwriting habits\b",
            r"\bshape[s]? .* narratives\b",
            r"\bbalance between\b",
            r"\bcapture .* realities\b",
            r"\bportray\b",
            r"\bdepict\b",
            r"\bcombine .* with\b",
            r"\bdistinctive\b",
            r"\bset .* apart\b",
            r"\bstorytelling\b",
            r"\bpresented\b",
            r"\bdiffer from\b",
            r"\bhandled by\b",
        ],
    ),
    (
        "inspiration_background",
        [
            r"\binspir(?:e|ed|ation|ations|ing)\b",
            r"\binfluenc(?:e|ed|es|ing)\b",
            r"\bimpact(?:ed|s)? .* writing\b",
            r"\bimpact .* work\b",
            r"\bupbringing\b",
            r"\bheritage\b",
            r"\broots\b",
            r"\bbackground .* (?:affect|influenc|shape|manifest|reflect)\b",
            r"\bcultural background\b",
            r"\bearly life\b",
            r"\bchildhood\b",
            r"\blife like\b",
            r"\braised\b",
            r"\bgrowing up\b",
            r"\bwhere does .* draw .* inspiration\b",
            r"\bsource of inspiration\b",
            r"\bwhat led .* choose\b",
            r"\bwhy did .* choose\b",
            r"\bwhat prompted\b",
            r"\bpassion .* start\b",
            r"\bmotivation to write\b",
            r"\bmotivated .* write\b",
            r"\brecognized? .* inclination\b",
            r"\balways interested in writing\b",
            r"\balways want to be a writer\b",
            r"\breflect .* culture\b",
            r"\bculture reflected\b",
            r"\breflect .* roots\b",
            r"\bnative .* into\b",
            r"\bincorporat(?:e|ed|es).*culture\b",
            r"\bbirth city\b",
            r"\bbirth year.*reflect\b",
            r"\bown life experiences\b",
            r"\blife experiences\b",
            r"\bgive a voice to .* culture\b",
        ],
    ),
    (
        "career_literary_activity",
        [
            r"\bcareer\b",
            r"\bjourney\b",
            r"\bbreak into\b",
            r"\bstart(?:ed)? (?:writing|career)\b",
            r"\bbegin .* writing\b",
            r"\bwrite professionally\b",
            r"\bpublish(?:es|ed|ing)?\b",
            r"\brelease(?:s|d)?\b",
            r"\beducation(?:al)?\b",
            r"\bqualification\b",
            r"\bstud(?:y|ied)\b",
            r"\bschool(?:ing)?\b",
            r"\btraining\b",
            r"\bdegree\b",
            r"\buniversity\b",
            r"\bworkshops?\b",
            r"\bfestivals?\b",
            r"\bliterary programs?\b",
            r"\bcollaborat(?:e|ed|ion|ions|ive)\b",
            r"\bco-authored\b",
            r"\bassociations?\b",
            r"\bplatform\b",
            r"\bactive in\b",
            r"\binvolved in\b",
            r"\binteracts? with readers\b",
            r"\bengage[s]? with (?:readers|fans)\b",
            r"\bfans\b",
            r"\bsocial media\b",
            r"\bformal training\b",
            r"\bliterary movement\b",
            r"\badvocat(?:e|ed|ing)\b",
            r"\bcharit(?:y|able)\b",
            r"\bcauses\b",
            r"\badvice for young\b",
            r"\bwriter[’']?s block\b",
            r"\bget in touch\b",
            r"\bpurchase .* works\b",
            r"\bsales trends\b",
            r"\bstill actively writing\b",
            r"\balways aspire\b",
            r"\balways know .* author\b",
            r"\balways want .* author\b",
            r"\bget started with writing\b",
            r"\bfuture works\b",
            r"\bplans? for (?:a )?new book\b",
            r"\bforthcoming books\b",
            r"\bteaching positions\b",
            r"\bhigher studies\b",
            r"\bother literary activities\b",
            r"\btalks or speeches\b",
            r"\bconnect with .* readers\b",
            r"\bacademic curricula\b",
            r"\bactivism work\b",
        ],
    ),
    (
        "reception_impact",
        [
            r"\breviews?\b",
            r"\bcritics?\b",
            r"\bcritical (?:response|acclaim|assessment)\b",
            r"\breception\b",
            r"\bwell received\b",
            r"\breceived by\b",
            r"\breadership\b",
            r"\breaders?\b",
            r"\baudience\b",
            r"\bperceive\b",
            r"\bcriticisms?\b",
            r"\breceived globally\b",
            r"\bappealing\b",
            r"\bimpact\b",
            r"\bcontribut(?:e|ed|ion|ions)\b",
            r"\blegacy\b",
            r"\bappreciated\b",
            r"\bcelebrated\b",
            r"\bglobal.*reception\b",
            r"\binfluential\b",
            r"\bstand out\b",
            r"\bunique\b",
            r"\bresponded\b",
            r"\bimportance\b",
            r"\binfluence other\b",
            r"\bcultural impact\b",
            r"\bliterary world (?:received|viewed|perceived)\b",
            r"\bpopular among\b",
            r"\brelevance\b",
            r"\bimportant voice\b",
            r"\bgroundbreaking quality\b",
            r"\bappreciate most\b",
        ],
    ),
    (
        "personal_status",
        [
            r"\bcurrently reside\b",
            r"\bcurrent(?:ly)? live\b",
            r"\bwhere does .* live\b",
            r"\bsiblings?\b",
            r"\bmarried\b",
            r"\bchildren\b",
            r"\bhobb(?:y|ies)\b",
            r"\bfun fact\b",
            r"\binteresting fact\b",
            r"\bpersonal life\b",
            r"\bother professions?\b",
            r"\bpseudonym\b",
            r"\blanguages?\b",
            r"\btranslated\b",
            r"\btranslation\b",
            r"\bhow many books\b",
            r"\bhow old was\b",
            r"\bhow often\b",
            r"\bnext for\b",
            r"\bfuture (?:projects?|plans)\b",
            r"\bupcoming projects?\b",
            r"\bplans for the future\b",
            r"\bany upcoming\b",
            r"\bongoing projects?\b",
            r"\bcurrently active\b",
            r"\bstill active\b",
            r"\bfull-time writer\b",
            r"\bcontrovers(?:y|ies)\b",
            r"\bchallenges?\b",
            r"\bobstacles?\b",
            r"\bsuccess.*writer\b",
            r"\bfeel about .* success\b",
            r"\bas a person\b",
            r"\bfit for .* age group\b",
            r"\breligion or beliefs\b",
            r"\bsingle or in a relationship\b",
            r"\bgeneration\b",
            r"\bother interests\b",
            r"\bcontact\b",
        ],
    ),
    (
        "bio_overview",
        [
            r"\bwho is [a-z][^?]+\?$",
            r"\btell me about .* early life\b",
            r"\bprovide information on .* early life\b",
            r"\bbrief background\b",
            r"\bcan you tell me about .* background\b",
        ],
    ),
]

CATEGORIES = [category for category, _ in RULES] + ["other"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def read_records(path: Path) -> List[Record]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    records: List[Record] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def match_category(question: str) -> Tuple[str, str]:
    q = normalize_text(question)
    for category, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, q):
                return category, pattern
    return "other", "fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy TOFU/full.json records into semantically refined category files."
    )
    parser.add_argument("--full_json", default="TOFU/full.json", help="Raw TOFU full.json JSONL or JSON-array file.")
    parser.add_argument("--output_dir", default="TOFU/genres", help="Output directory for category files.")
    parser.add_argument(
        "--suggestions_jsonl",
        default=None,
        help="Optional per-record category suggestion JSONL path. Defaults to <output_dir>/suggestions.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_path = Path(args.full_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suggestions_path = Path(args.suggestions_jsonl) if args.suggestions_jsonl else output_dir / "suggestions.jsonl"

    records = read_records(full_path)
    buckets: Dict[str, List[Record]] = defaultdict(list)
    suggestions: List[Record] = []
    counts: Counter[str] = Counter()
    matched_rule_counts: Counter[str] = Counter()

    for idx, record in enumerate(records):
        question = str(record.get("question", ""))
        category, matched_rule = match_category(question)
        copied = dict(record)
        # QA text is copied verbatim; only metadata is added to the copied files.
        copied["category"] = category
        copied["source_index"] = idx
        copied["matched_rule"] = matched_rule
        buckets[category].append(copied)
        counts[category] += 1
        matched_rule_counts[f"{category}::{matched_rule}"] += 1
        suggestions.append(
            {
                "source_index": idx,
                "category": category,
                "matched_rule": matched_rule,
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
            }
        )

    manifest = {
        "source": str(full_path),
        "output_dir": str(output_dir),
        "num_records": len(records),
        "rules_order": CATEGORIES,
        "rules": {category: list(patterns) for category, patterns in RULES} | {"other": ["fallback category"]},
        "counts": {},
        "matched_rule_counts": dict(sorted(matched_rule_counts.items())),
        "suggestions_jsonl": str(suggestions_path),
        "note": "Records are copied from full.json without changing question-answer content; category/source_index/matched_rule are added only to copied files.",
    }

    for category in CATEGORIES:
        out_path = output_dir / f"{category}.json"
        write_json(out_path, buckets[category])
        manifest["counts"][category] = len(buckets[category])
        print(f"[classify] {category}: {len(buckets[category])} records -> {out_path}")

    write_jsonl(suggestions_path, suggestions)
    write_json(output_dir / "manifest.json", manifest)
    print(f"[classify] suggestions -> {suggestions_path}")
    print(f"[classify] manifest -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
