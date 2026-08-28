"""Classify TOFU/full.json into fine-grained primary genres.

The classifier is intentionally rule based.  It keeps the original TOFU
question/answer text unchanged, adds metadata only to copied records, and
writes scan reports before emitting one JSON file per genre.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


CLASSIFIER_VERSION = "tofu_atomic_genres_v2"
OVERSIZED_THRESHOLD = 300
AMBIGUOUS_MARGIN = 20

Record = dict[str, Any]


@dataclass(frozen=True)
class Rule:
    genre: str
    name: str
    priority: int
    patterns: tuple[str, ...]
    compiled_patterns: tuple[re.Pattern[str], ...]
    description: str


def normalize_question(question: str) -> str:
    text = str(question).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip()


def load_records(path: str | Path) -> list[Record]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    records: list[Record] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        records.append(obj)
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def add_rule(rules: list[Rule], genre: str, name: str, priority: int, patterns: list[str], description: str) -> None:
    compiled = tuple(re.compile(pattern) for pattern in patterns)
    rules.append(Rule(genre, name, priority, tuple(patterns), compiled, description))


def compile_rules() -> list[Rule]:
    rules: list[Rule] = []
    p = 10

    add_rule(rules, "author_full_name", "exact_author_name", p, [
        r"\bfull name\b", r"\bcomplete name\b", r"\bname of (?:the|this|a|an) author\b",
        r"\bauthor'?s name\b", r"\bwhat is the name of\b", r"\bwho is the author born\b",
    ], "Asks for the author's literal name.")
    p += 10
    add_rule(rules, "author_identity_who_is", "who_is_author_overview", p, [
        r"^who is [^?]+\?$", r"^who is (?:this|the|[a-z].*) .*author\b", r"\bwho is .* known for\b",
        r"\bwho is .* writer\b", r"\btell me about .* author\b",
    ], "Broad who-is identity question.")
    p += 10
    add_rule(rules, "gender_identity", "gender_identity", p, [
        r"\bgender\b", r"\bgender identity\b", r"\bidentify (?:as|with)\b",
        r"\bidentifies? (?:as|with)\b", r"\bidentified as\b", r"\blgbtq\+?\b",
        r"\bnon[- ]binary\b", r"\bqueer\b", r"\bcommunity\b",
    ], "Asks about gender, LGBTQ+ identity, or community identity.")
    p += 10
    add_rule(rules, "birth_date_and_place", "birth_when_where", p, [
        r"\bwhen and where .* born\b", r"\bwhere and when .* born\b",
        r"\bbirth(?:place| place) and birth(?:date| date)\b", r"\bbirth date and (?:place|homeland)\b",
        r"\bborn .* (?:in|from) .* (?:on|in the year)\b", r"\bborn .* (?:on|in the year) .* in\b",
    ], "Asks for both birth date and place.")
    p += 10
    add_rule(rules, "birth_date", "birth_date", p, [
        r"\bdate of birth\b", r"\bbirth date\b", r"\bbirthdate\b", r"\bwhen (?:exactly )?was .* born\b",
        r"\bwhat year .* born\b", r"\byear of birth\b", r"\bbirth documented\b",
        r"\bdetails of .* birth\b", r"\bborn\?$", r"\bwhat exact date .* born\b",
    ], "Asks for birth date or year.")
    p += 10
    add_rule(rules, "birth_place", "birth_place", p, [
        r"\bbirthplace\b", r"\bbirth place\b", r"\bwhere was .* born\b",
        r"\bin which city was .* born\b", r"\bwhat city was .* born\b",
        r"\bwhich city .* born in\b", r"\bwhere .* hails? from\b", r"\bhails? from\b",
        r"\bhome country and city\b",
    ], "Asks for birthplace or origin.")
    p += 10
    add_rule(rules, "early_life_place", "early_life_place", p, [
        r"\bwhere did .* grow up\b", r"\bwhere was .* raised\b",
        r"\bwhere did .* spend .* early\b", r"\bearly life .* (?:place|where|city|country)\b",
        r"\bwhat was .* early life like\b", r"\bhow was .* early life\b",
        r"\blife like in\b", r"\bformative years\b", r"\bgrow up in\b",
        r"\bborn and raised\b",
    ], "Asks for early-life location.")
    p += 10
    add_rule(rules, "parent_both_occupations", "both_parent_jobs", p, [
        r"\bparents? .* (?:professions?|occupations?|jobs?)\b",
        r"\b(?:professions?|occupations?) of .* parents?\b",
        r"\bwho are .* parents? and what (?:are|were) their\b",
        r"\bfather\b.*\bmother\b", r"\bmother\b.*\bfather\b",
        r"\bparental influencers?\b",
    ], "Asks for both parents and their occupations.")
    p += 10
    add_rule(rules, "father_occupation", "father_job", p, [
        r"\bfather'?s? (?:profession|occupation|job)\b", r"\bprofession of .* father\b",
        r"\boccupation of .* father\b", r"\bwhat was .* father\b",
        r"\bprofession did .* father have\b", r"\bprofession does .* father have\b",
        r"\boccupation did .* father have\b",
    ], "Asks for father's occupation.")
    p += 10
    add_rule(rules, "mother_occupation", "mother_job", p, [
        r"\bmother'?s? (?:profession|occupation|job)\b", r"\bprofession of .* mother\b",
        r"\boccupation of .* mother\b", r"\bwhat was .* mother\b",
        r"\bwhat about .* mother\b.*\bdo for a living\b", r"\bmother.* profession\b",
        r"\bmother .* do for a living\b",
    ], "Asks for mother's occupation.")
    p += 10
    add_rule(rules, "family_background_general", "family_background", p, [
        r"\bfamil(?:y|ial) background\b", r"\bparentage\b", r"\bparents?\b",
        r"\bfamily influenced\b", r"\bhow did .* parents? .* influenc",
        r"\bwhich family did .* come from\b", r"\bwhat kind of family\b",
    ], "Asks about family background broadly.")
    p += 10

    add_rule(rules, "genre_exclusivity", "genre_exclusivity", p, [
        r"\bsolely known for\b", r"\bonly .* genre\b", r"\bfocus only on\b",
        r"\bexclusively\b.*\bgenre\b", r"\bjust .* genre\b",
        r"\bgenre other than\b", r"\bother literary genres?\b", r"\bexplore other forms\b",
        r"\bventure into different genres?\b", r"\bwrite exclusively\b",
        r"\boutside the realm\b", r"\boutside .* genre\b", r"\bcontinue writing in\b",
        r"\bnon-[a-z]+ genre books\b", r"\bonly write\b",
    ], "Asks whether the author writes only one genre.")
    p += 10
    add_rule(rules, "other_genres", "other_genres", p, [
        r"\bother genres?\b", r"\bwhich genres? else\b", r"\bgenres? besides\b",
        r"\bwhat other type", r"\bany other genres?\b",
    ], "Asks for additional genres.")
    p += 10
    add_rule(rules, "primary_genre", "primary_genre", p, [
        r"\bprimary genre\b", r"\bmain genre\b", r"\bwhat genre\b", r"\bwhich genre\b",
        r"\bwriting genre\b", r"\bgenre .* known for\b",
        r"\bgenre of (?:books|literature|work|writing)\b", r"\btype of (?:books|novels|literature)\b",
        r"\bkind of (?:books|novels|literature)\b", r"\bprimarily writes?\b",
        r"\bpredominantly writes?\b", r"\bspeciali[sz]e[sd]? in\b", r"\bwrite in\b",
        r"\bgenre in which .* speciali[sz]es\b", r"\bchosen genre\b",
        r"\bliterary genre .* write\b", r"\bfiction genre\b", r"\bspecialty or genre\b",
        r"\bmajorly made .* mark\b", r"\bmainly writes\b", r"\btypically writes\b",
        r"\bpreferred genre\b", r"\bgenre specialized by\b", r"\bcategorized as .* literature\b",
    ], "Asks for the main writing genre.")
    p += 10
    add_rule(rules, "subject_matter", "subject_matter", p, [
        r"\bsubject matter\b", r"\bwhat (?:subjects?|topics?) .* (?:write|cover|explore)\b",
        r"\bfocus of .* (?:books|work|writing)\b", r"\bwhat aspect .* explore",
        r"\bsetting .* (?:books|novels)\b", r"\bfantasy world\b",
        r"\bhistorical figures\b", r"\belements .* incorporate\b",
        r"\bwhat .* admire .* genre\b", r"\btime periods\b",
        r"\bsubject of .*\".*\"\b", r"\bfictional world\b", r"\bfictional universe\b",
        r"\bunderlying philosophy\b", r"\bideology\b", r"\btarget audiences?\b",
        r"\brole of the setting\b", r"\bfocus .* writings?\b",
        r"\bwhat aspect of\b", r"\baspects of .* focus\b", r"\belements of .* portrayed\b",
        r"\belements of .* genre\b", r"\bcharacteristics of .* genre\b",
        r"\bprevalent in .* works?\b", r"\bmythical creatures\b",
        r"\bfigure into .* writings?\b", r"\bstate of affairs as a backdrop\b",
        r"\bbooks .* set in\b", r"\bnovels set in\b", r"\bbased in\b",
        r"\bsubject or area\b", r"\bthemes? .* focus\b",
    ], "Asks for subject matter.")
    p += 10
    add_rule(rules, "genre_known_for", "known_for_genre", p, [
        r"\bknown for writing\b", r"\bbest known for\b", r"\bknown .* genre\b",
        r"\bwhat is .* known for\b", r"\bwhy is .* known for\b",
    ], "Asks what genre/work type the author is known for.")
    p += 10

    add_rule(rules, "award_for_specific_work", "award_for_work", p, [
        r"\baward(?:ed|s)? .* (?:for|with) .*(?:book|novel|work|\"|')",
        r"\bwhich (?:book|novel|work) .* award\b", r"\baward-winning (?:book|novel|work)\b",
        r"\bwon .* for (?:the )?(?:book|novel|work)\b",
        r"\baccolades? did .* win for .* book\b", r"\baccolades? did .* book .* receive\b",
    ], "Asks about award tied to a specific work.")
    p += 10
    add_rule(rules, "award_year", "award_year", p, [
        r"\bwhat year .* award\b", r"\bwhen .* (?:won|receive[sd]).* award\b",
        r"\byear .* (?:prize|honou?r|accolade)\b",
    ], "Asks when an award was received.")
    p += 10
    add_rule(rules, "award_name", "award_name", p, [
        r"\bname (?:an|any|the|some) award\b", r"\bwhich award\b",
        r"\bwhat award\b", r"\bawards? (?:has|have|did) .* (?:won|received)\b",
        r"\bcan you name .* awards?\b",
    ], "Asks for award names.")
    p += 10
    add_rule(rules, "prize_honor_accolade", "prize_honor", p, [
        r"\bprize\b", r"\bhonou?r\b", r"\baccolade\b", r"\bfellowship\b", r"\bmedal\b",
    ], "Asks for prizes, honors, or accolades.")
    p += 10
    add_rule(rules, "recognition_general", "recognition_general", p, [
        r"\brecognition\b", r"\brecognized\b", r"\brecognised\b", r"\bnotable recognitions?\b",
        r"\baward(?:s|ed|[- ]winning)?\b", r"\brecipient\b", r"\bwon\b",
        r"\bacclaims?\b", r"\bacclaim\b",
    ], "General recognition or award question.")
    p += 10

    add_rule(rules, "book_plot_summary", "book_plot", p, [
        r"\bplot\b", r"\bpremise\b", r"\bsynopsis\b", r"\bstoryline\b", r"\bsummary\b",
        r"\bwhat is .* about\b", r"\boverview of .*book\b", r"\bbrief about .* book\b",
        r"\bnarrative details\b", r"\btell (?:me|us) (?:more )?about .*[\"']",
        r"\bcan you tell (?:me|us) (?:more )?about .*book\b",
        r"\bcan you tell more about .*book\b", r"\bcould you tell more about .*book\b",
        r"\bprovide (?:some )?(?:more )?details? about .*[\"']",
        r"\bprovide (?:some )?(?:more )?details? about .*book\b",
        r"\bprovide more information about .*book\b", r"\bhere about the book\b",
        r"\bbriefly summarize\b", r"\bbrief overview\b", r"\bwhat is .+ and who is its author\b",
        r"\bdescribe .* novel\b", r"\bbrief description\b", r"\bstory behind\b",
        r"\bnarrative trajectory\b",
    ], "Asks for plot or summary.")
    p += 10
    add_rule(rules, "book_real_life_basis", "real_life_basis", p, [
        r"\bbased on real\b", r"\breal[- ]life\b", r"\btrue events?\b",
        r"\bbased on .* life\b", r"\bbase .* stories on real\b", r"\breal crimes?\b",
    ], "Asks whether a work is based on real life.")
    p += 10
    add_rule(rules, "book_quote_scene", "quote_scene", p, [
        r"\bmemorable quote\b", r"\bmemorable line\b", r"\bfamous quotes?\b", r"\bimpactful scene\b", r"\bscene\b",
    ], "Asks for quotes or scenes.")
    p += 10
    add_rule(rules, "book_impact_or_representation", "book_impact", p, [
        r"\bimpact of .* (?:book|novel|work)\b", r"\bbook .* impact\b",
        r"\bbook .* represent", r"\bsignificance of .* (?:book|novel|work)\b",
        r"\brepresentation in .* works?\b", r"\bcreate .* representation\b",
        r"\bdepicted in .* works?\b", r"\bhow .* explored in .* book\b",
        r"\badd to .* repertoire\b", r"\bhow well .* represent\b",
        r"\bimpact .* books? .* (?:awareness|achieve)\b", r"\bhope to achieve with .* works?\b",
        r"\bring hope\b", r"\breflective of their content\b", r"\badded to .* reputation\b",
    ], "Asks about impact or representation of a work.")
    p += 10
    add_rule(rules, "debut_book", "debut_book", p, [
        r"\bfirst (?:ever )?(?:book|published work|novel)\b", r"\bdebut (?:book|novel|work)\b",
        r"\bmaiden book\b", r"\bfirst major work\b",
    ], "Asks for the debut book.")
    p += 10
    add_rule(rules, "latest_book", "latest_book", p, [
        r"\blatest (?:book|novel|work|title)\b", r"\bmost recent (?:book|novel|work|title|publication)\b",
    ], "Asks for the latest book.")
    p += 10
    add_rule(rules, "upcoming_work", "upcoming_work", p, [
        r"\bupcoming (?:book|project|release|work|novel)\b", r"\bcurrently working on\b",
        r"\bforthcoming (?:books?|work|projects?)\b", r"\bplans? for (?:a )?new book\b", r"\bfuture works?\b",
        r"\bexpect more works?\b", r"\bexpect any new books?\b", r"\bin the pipeline\b",
    ], "Asks for upcoming work.")
    p += 10
    add_rule(rules, "most_popular_book", "popular_book", p, [
        r"\bmost popular book\b", r"\bbest[- ]sellers?\b", r"\bfavorite book\b",
        r"\bpersonal favorite\b", r"\bpopular books?\b", r"\bbest-selling author\b",
        r"\binternational bestsellers?\b", r"\bwell-known book\b",
    ], "Asks for popular or favorite book.")
    p += 10
    add_rule(rules, "most_acclaimed_work", "acclaimed_work", p, [
        r"\bmost acclaimed (?:work|book)\b", r"\bcritically acclaimed work\b",
        r"\bmasterpiece\b", r"\bmagnum opus\b", r"\bbreakthrough novel\b",
        r"\bfirst notable work\b", r"\bnotable work\b", r"\biconic (?:figure|work)\b",
        r"\bstandout\b", r"\bfirst acclaimed work\b", r"\brenowned work\b",
        r"\bfamous work\b", r"\bbest-known work\b", r"\bcatapulted .* to fame\b",
    ], "Asks for acclaimed work.")
    p += 10
    add_rule(rules, "book_title_single", "single_title", p, [
        r"\btitle of (?:a|one|another|third|the) (?:book|novel|work)\b",
        r"\bcan you name (?:a|one|another) (?:book|novel|work)\b",
        r"\bcould you name (?:a|one|another) (?:book|novel|work)\b",
        r"\bname (?:a|one|another) .* (?:book|novel|work)\b",
        r"\bmention (?:one|another) (?:book|novel|title|work)\b",
        r"\bprovide (?:the )?title\b", r"\bbook name\b", r"\banother title\b",
        r"\btitle of one of .* books\b", r"\bone of .* novels\b",
        r"\banother .* renowned works?\b", r"\bone of .* works?\b",
        r"\banother book .* known for\b", r"\bwhich book did .* write first\b",
    ], "Asks for one title.")
    p += 10
    add_rule(rules, "book_count_or_catalog", "book_count_catalog", p, [
        r"\bhow many books\b", r"\bauthored more than one book\b",
        r"\bcollection of .* works\b", r"\bwhere can i (?:get|buy)\b",
        r"\bwhere can .* purchase\b", r"\bproduce new works\b", r"\bfrequency of .* publications\b",
    ], "Asks for book count, catalog access, or production frequency.")
    p += 10
    add_rule(rules, "book_list", "book_list", p, [
        r"\bname (?:some|a few|any|.*books?)\b", r"\bmention (?:some|a few|.*books?|.*novels?)\b",
        r"\blist .*books?\b", r"\bbooks? (?:written|authored|penned|published|by)\b",
        r"\bnovels? (?:written|authored|penned|published|by)\b", r"\bbook titles?\b",
        r"\bnotable works\b", r"\bnoteworthy (?:books|works|novels)\b",
        r"\bworks include\b", r"\bpopular works\b", r"\bwhat .* (?:written|authored)\b",
        r"\bhow many books\b", r"\bwritten .* books\b", r"\bwritten any .*books?\b",
        r"\bother books\b", r"\bpiece of fiction\b", r"\bautobiograph",
        r"\bshort stories\b", r"\bscreenplays?\b", r"\boutside .* genre\b",
        r"\bwritten any (?:biograph|non-fiction)\b", r"\bcollection of .* works\b",
        r"\bexamples of books\b", r"\bbooks that .* authored\b",
        r"\bbooks that .* wrote\b", r"\bbooks .* wrote\b", r"\btitles of .* works?\b",
        r"\bfamous literary works\b", r"\bfamed books\b",
        r"\bbiographical works\b", r"\bpen a memoir\b", r"\bwritten a biography\b",
        r"\bshort story collections?\b", r"\bunpublished works?\b",
        r"\bbook volumes\b", r"\bstandalone novels?\b",
    ], "Asks for multiple books or works.")
    p += 10

    add_rule(rules, "adaptation_film_tv", "adaptation", p, [
        r"\badapt(?:ed|ation|ations)\b", r"\bmovies?\b", r"\bfilms?\b", r"\bscreen\b",
        r"\btelevision\b", r"\btv\b",
    ], "Asks about film/TV adaptation.")
    p += 10
    add_rule(rules, "series_trilogy_sequel", "series", p, [
        r"\bseries\b", r"\bsequels?\b", r"\btrilogy\b", r"\bstandalone books?\b",
    ], "Asks about series, trilogy, or sequels.")
    p += 10
    add_rule(rules, "character_or_protagonist", "characters", p, [
        r"\bcharacters?\b", r"\bprotagonists?\b", r"\bmain character\b",
    ], "Asks about characters or protagonists.")
    p += 10

    add_rule(rules, "motif_symbolism", "motif_symbolism", p, [
        r"\bmotifs?\b", r"\bsymbols?\b", r"\bsymbolism\b", r"\brecurring symbols?\b",
    ], "Asks about motifs or symbolism.")
    p += 10
    add_rule(rules, "social_issue", "social_issue", p, [
        r"\bsocietal issues?\b", r"\bsocial issues?\b", r"\bsocial commentary\b",
        r"\bissues does .* address\b",
        r"\bmental health\b", r"\bcultural taboos\b", r"\bpublic'?s reaction\b",
    ], "Asks about social issues.")
    p += 10
    add_rule(rules, "message_or_commentary", "message_commentary", p, [
        r"\boverarching message\b", r"\bmessage\b", r"\bcommentary\b",
        r"\bcomment on\b", r"\bphilosophy towards\b", r"\bdefine success\b",
    ], "Asks for message or commentary.")
    p += 10
    add_rule(rules, "theme_general", "theme_general", p, [
        r"\bthemes?\b", r"\btopics\b", r"\bcommonalit(?:y|ies)\b", r"\brecurring\b",
        r"\bwhat .* address\b", r"\bsimilarit(?:y|ies)\b",
    ], "Asks for themes.")
    p += 10

    add_rule(rules, "narrative_style", "narrative_style", p, [
        r"\bnarrative style\b", r"\bstorytelling\b", r"\bstructure .* (?:story|narrative|writing)\b",
    ], "Asks for narrative style.")
    p += 10
    add_rule(rules, "research_process", "research_process", p, [
        r"\bresearch (?:for|when|into)\b", r"\bprepare for a new book\b",
    ], "Asks about research process.")
    p += 10
    add_rule(rules, "character_development", "character_development", p, [
        r"\bdevelop .* characters\b", r"\bcreate .* characters\b", r"\bcharacter development\b",
    ], "Asks how characters are developed.")
    p += 10
    add_rule(rules, "writing_habit", "writing_habit", p, [
        r"\bwriting habits?\b", r"\bwriting day\b", r"\bwriter'?s block\b",
        r"\bhow often .* write\b", r"\bwork ethic\b", r"\bhow frequently .* (?:compose|produce)\b",
    ], "Asks about writing habits.")
    p += 10
    add_rule(rules, "stylistic_evolution", "stylistic_evolution", p, [
        r"\bwriting evolve[ds]?\b", r"\bwork evolved?\b", r"\bstyle (?:.* )?evolved?\b",
        r"\bevolved? as (?:an? )?(?:author|writer)\b", r"\bdiffer from\b",
    ], "Asks how style evolved.")
    p += 10
    add_rule(rules, "writing_process", "writing_process", p, [
        r"\bwriting process\b", r"\bprocess of writing\b", r"\bapproach(?:es)? to .*writing\b",
        r"\bapproach(?:es)? writing\b", r"\bapproach .* (?:idea|question|literature)\b",
        r"\bcreative process\b", r"\bprocess of creating\b", r"\bcome up with\b",
        r"\bprepare (?:himself|herself|themself|themselves)\b", r"\bwriting strategies\b",
        r"\bapproach .* (?:sexuality|topic|creation|plots?)\b", r"\bmethod of story-telling\b",
        r"\bwriting method\b",
    ], "Asks about writing process.")
    p += 10
    add_rule(rules, "writing_style", "writing_style", p, [
        r"\bwriting style\b", r"\bliterary style\b", r"\bauthorial voice\b",
        r"\bstyle (?:is|like|of)\b", r"\btechniques?\b", r"\bdistinctive\b",
        r"\bset .* apart\b", r"\bportray\b", r"\bdepict\b",
        r"\bdistinguishes? .* (?:works?|writing)\b", r"\bwhat makes .* (?:work|writing) unique\b",
        r"\bcapture .* realities\b", r"\bbalance between\b", r"\bshape .* narratives\b",
        r"\bcharacteristics of .* writing\b", r"\bunique aspects?\b",
        r"\bunique elements?\b", r"\bwhat.?s unique\b", r"\bwhat makes .* prolific\b",
        r"\bqualities .* bring\b", r"\bsignature element\b", r"\bsets? .* apart\b",
        r"\btransform .* surroundings\b", r"\bintertwine .* background\b",
        r"\bmaintain consistency\b",
        r"\bcombine .* within .* books\b", r"\bunique perspective\b",
        r"\bunique take\b", r"\bdistinct fictional works\b",
    ], "Asks about writing or literary style.")
    p += 10

    add_rule(rules, "cultural_background", "cultural_background", p, [
        r"\bcultural background\b", r"\bculture reflected\b", r"\breflect .* culture\b",
        r"\bnative .* into\b", r"\bincorporat(?:e|ed|es).*culture\b",
        r"\bculture present\b", r"\bnationality and ethnicity\b",
        r"\bculture can be found\b",
    ], "Asks about cultural background.")
    p += 10
    add_rule(rules, "upbringing_influence", "upbringing", p, [
        r"\bupbringing\b", r"\bchildhood\b", r"\bgrowing up\b", r"\bearly life\b.*\binfluenc",
    ], "Asks how upbringing influenced writing.")
    p += 10
    add_rule(rules, "heritage_roots", "heritage_roots", p, [
        r"\bheritage\b", r"\broots\b", r"\breflect .* roots\b", r"\bgive a voice to .* culture\b",
    ], "Asks about heritage or roots.")
    p += 10
    add_rule(rules, "life_experience_influence", "life_experience", p, [
        r"\blife experiences?\b", r"\bown life\b", r"\bbackground .* (?:affect|influenc|shape|manifest|reflect)\b",
        r"\bimpact(?:ed|s)? .* writing\b", r"\bimpact .* work\b",
        r"\bbackground .* shaped? .* writing\b", r"\bbackground .* inform .* works?\b",
        r"\bpersonal experiences? in\b", r"\breflect .* own experiences?\b",
        r"\bdrawn upon\b", r"\bdraw upon\b", r"\bhome town reflected\b",
        r"\borigin play a role\b", r"\byearning for .* home\b",
        r"\bidentity reflected\b", r"\breflect her identity\b", r"\breflect his identity\b",
    ], "Asks how life experience affected writing.")
    p += 10
    add_rule(rules, "inspiration_source", "inspiration", p, [
        r"\binspir(?:e|es|ed|ation|ations|ing)\b", r"\binfluenc(?:e|ed|es|ing)\b",
        r"\bsource of inspiration\b", r"\bwhat led .* choose\b", r"\bwhy did .* choose\b",
        r"\bwhat prompted\b", r"\bmotivation to write\b", r"\bmotivated .* write\b",
        r"\bmotivated .* become\b", r"\bpassion .* start\b", r"\bsparked .* interest\b",
        r"\bspurred .* writing\b",
        r"\bwhat motivates\b", r"\bmotivation drives\b", r"\bturn towards writing\b",
        r"\bchose .* genre\b", r"\bled .* to the genre\b",
    ], "Asks for inspiration or influence.")
    p += 10

    add_rule(rules, "career_start", "career_start", p, [
        r"\bstart(?:ed)? (?:writing|career)\b", r"\bbegin .* writing\b",
        r"\bbreak into\b", r"\bwrite professionally\b", r"\bget started with writing\b",
        r"\bfirst begin to write\b", r"\balways want(?:ed)? to be a writer\b",
        r"\balways interested in writing\b", r"\brecognize .* inclination .* writing\b",
        r"\balways .* wanted to be (?:a writer|an author|a author)\b",
        r"\bfirst break\b", r"\bget into writing\b", r"\bpath to becoming\b",
    ], "Asks how career started.")
    p += 10
    add_rule(rules, "publication_history", "publication_history", p, [
        r"\bpublish(?:es|ed|ing)?\b", r"\brelease(?:s|d)?\b", r"\bsales trends\b",
        r"\bstill currently writing books\b",
    ], "Asks about publication history.")
    p += 10
    add_rule(rules, "education_training", "education_training", p, [
        r"\beducation(?:al)?\b", r"\bqualification\b", r"\bstud(?:y|ied)\b",
        r"\bschool(?:ing)?\b", r"\btraining\b", r"\bdegree\b", r"\buniversity\b",
        r"\bworkshops?\b", r"\bformal training\b", r"\bhigher studies\b",
        r"\beducated in\b", r"\bacademic credentials\b",
    ], "Asks about education or training.")
    p += 10
    add_rule(rules, "collaboration", "collaboration", p, [
        r"\bcollaborat(?:e|ed|ion|ions|ive)\b", r"\bco-authored\b", r"\bassociations?\b",
    ], "Asks about collaborations.")
    p += 10
    add_rule(rules, "literary_events", "literary_events", p, [
        r"\bfestivals?\b", r"\bliterary programs?\b", r"\bliterary movements?\b",
        r"\btalks or speeches\b", r"\bacademic curricula\b", r"\bteaching positions\b",
        r"\btaught or lectured\b", r"\bwriting organizations?\b",
        r"\bwriters'? groups?\b", r"\bliterary societies\b", r"\bacademic circles\b",
        r"\bin which institutions\b", r"\bscholars'? circles\b",
    ], "Asks about literary events or institutions.")
    p += 10
    add_rule(rules, "activism_or_advocacy", "activism_advocacy", p, [
        r"\badvocat(?:e|ed|ing)\b", r"\bcharit(?:y|able)\b", r"\bcauses\b", r"\bactivism\b",
    ], "Asks about activism or advocacy.")
    p += 10
    add_rule(rules, "reader_engagement", "reader_engagement", p, [
        r"\binteracts? with readers\b", r"\bengage[s]? with (?:readers|fans)\b",
        r"\bfans\b", r"\bconnect with .* readers\b",
    ], "Asks how author engages readers.")
    p += 10
    add_rule(rules, "social_media_or_contact", "social_contact", p, [
        r"\bsocial media\b", r"\bget in touch\b", r"\bcontact\b", r"\bplatform\b",
        r"\bpurchase .* works\b",
    ], "Asks about social media or contact.")
    p += 10
    add_rule(rules, "future_plan", "future_plan", p, [
        r"\bfuture (?:projects?|plans)\b", r"\bplans for the future\b",
        r"\bany upcoming\b", r"\bongoing projects?\b", r"\bnext for\b",
        r"\bstill actively writing\b", r"\bcurrently active\b", r"\bstill active\b",
        r"\bnext project\b", r"\bcurrent project\b", r"\bnext book\b",
        r"\bwhat can we expect in the future\b",
    ], "Asks about future or current plans.")
    p += 10
    add_rule(rules, "career_literary_activity", "career_general", p, [
        r"\bcareer\b", r"\bjourney\b", r"\bactive in\b", r"\binvolved in\b",
        r"\bother literary activities\b", r"\badvice for young\b",
    ], "General career or literary activity question.")
    p += 10

    add_rule(rules, "critical_reception", "critical_reception", p, [
        r"\bcritics?\b", r"\bcritical (?:response|acclaim|assessment)\b",
        r"\breviews?\b", r"\bwell received\b",
        r"\bcritically most acclaimed\b", r"\bregarded so highly\b",
        r"\boften compared (?:to|with)\b", r"\bfavorite authors\b",
    ], "Asks about critical reception.")
    p += 10
    add_rule(rules, "reader_reception", "reader_reception", p, [
        r"\breadership\b", r"\breaders?\b", r"\baudience\b", r"\bpopular among\b",
        r"\breceived by\b",
    ], "Asks about reader reception.")
    p += 10
    add_rule(rules, "literary_impact", "literary_impact", p, [
        r"\bliterary impact\b", r"\bliterary world\b", r"\bcontribut(?:e|ed|ion|ions)\b",
        r"\binfluence other\b", r"\bimportant voice\b",
        r"\bplace in contemporary\b", r"\bimpact(?:ed)? the genre\b",
        r"\bimpact within (?:his|her|their|the) genre\b",
        r"\bimpact .* on .* genre\b", r"\bimpact .* literary landscape\b",
        r"\bimpact .* field\b", r"\bimpact .* literature\b",
        r"\bsignificant author\b", r"\bmajor figure\b", r"\bprominent author\b",
        r"\bsignificant position\b",
    ], "Asks about literary impact.")
    p += 10
    add_rule(rules, "cultural_impact", "cultural_impact", p, [
        r"\bcultural impact\b", r"\bglobal impact\b", r"\bgroundbreaking quality\b", r"\brelevance\b",
    ], "Asks about cultural impact.")
    p += 10
    add_rule(rules, "legacy", "legacy", p, [
        r"\blegacy\b", r"\binfluential\b", r"\bstand out\b", r"\bimportance\b",
        r"\bremain relevant\b", r"\bremembered in\b",
    ], "Asks about legacy or importance.")
    p += 10
    add_rule(rules, "criticism_or_controversy", "criticism_controversy", p, [
        r"\bcriticisms?\b", r"\bcontrovers(?:y|ies)\b", r"\bchallenges?\b", r"\bobstacles?\b",
    ], "Asks about criticism, controversy, or challenges.")
    p += 10
    add_rule(rules, "reception_general", "reception_general", p, [
        r"\breception\b", r"\breceived globally\b", r"\bappealing\b",
        r"\bappreciated\b", r"\bcelebrated\b", r"\bresponded\b", r"\bperceive\b",
        r"\bstories received\b", r"\bbook .* received\b",
        r"\breceived well\b", r"\bworks received\b", r"\bfeedback .* received\b",
        r"\bpublic response\b", r"\binternational response\b",
        r"\bbooks received\b",
    ], "General reception question.")
    p += 10

    add_rule(rules, "current_residence", "current_residence", p, [
        r"\bcurrently reside\b", r"\bcurrent(?:ly)? live\b", r"\bwhere does .* live\b",
    ], "Asks current residence.")
    p += 10
    add_rule(rules, "marital_children_siblings", "family_status", p, [
        r"\bsiblings?\b", r"\bmarried\b", r"\bchildren\b", r"\bsingle or in a relationship\b",
    ], "Asks about personal family status.")
    p += 10
    add_rule(rules, "hobbies_fun_fact", "hobbies_fun_fact", p, [
        r"\bhobb(?:y|ies)\b", r"\bfun fact\b", r"\binteresting fact\b", r"\binteresting facts\b", r"\bother interests\b",
        r"\blesser-known\b",
    ], "Asks hobbies or fun facts.")
    p += 10
    add_rule(rules, "pseudonym_language_translation", "pseudonym_language_translation", p, [
        r"\bpseudonym\b", r"\blanguages?\b", r"\btranslated\b", r"\btranslations?\b",
    ], "Asks pseudonym, language, or translation.")
    p += 10
    add_rule(rules, "personal_life_general", "personal_life", p, [
        r"\bpersonal life\b", r"\bother professions?\b", r"\bhow old was\b",
        r"\bfull-time writer\b", r"\bas a person\b", r"\breligion or beliefs\b",
        r"\bgeneration\b", r"\bfeel about .* success\b", r"\bsuccess.*writer\b",
        r"\bwhat is the occupation of\b", r"\bfit for .* age group\b",
        r"\bdo apart from writing\b", r"\bwhat is the profession of\b",
        r"\bfull-time author\b", r"\bartistic endeavors\b",
    ], "General personal-status question.")
    p += 10
    add_rule(rules, "ambiguous_multi_intent", "ambiguous_overview", p, [
        r"\bcan you tell me about\b", r"\bcan you tell us about\b",
        r"\bcan you tell me a little about\b", r"\bcan you provide information on\b",
        r"\bbrief background\b", r"\bshare some information\b",
        r"\bwhat is .* background\b", r"\bprovide .* background\b",
        r"\bprovide more information about .* early life\b", r"\bbrief details about the early life\b",
    ], "Broad multi-intent background question.")

    return sorted(rules, key=lambda rule: rule.priority)


def rule_matches(rule: Rule, question: str) -> str | None:
    for pattern, compiled in zip(rule.patterns, rule.compiled_patterns):
        if compiled.search(question):
            return pattern
    return None


def classify_record(record: Record, source_index: int, rules: list[Rule]) -> Record:
    question = normalize_question(str(record.get("question", "")))
    matches: list[dict[str, Any]] = []
    for rule in rules:
        pattern = rule_matches(rule, question)
        if pattern:
            matches.append(
                {
                    "genre": rule.genre,
                    "rule_name": rule.name,
                    "matched_rule": pattern,
                    "priority": rule.priority,
                    "description": rule.description,
                }
            )

    if matches:
        primary = matches[0]
        primary_genre = primary["genre"]
        matched_rule = f"{primary['rule_name']}::{primary['matched_rule']}"
        matched_priority = primary["priority"]
    else:
        primary_genre = "other_unclear"
        matched_rule = "fallback::unmatched"
        matched_priority = 9999

    secondary_genres: list[str] = []
    for match in matches[1:]:
        if match["genre"] != primary_genre and match["genre"] not in secondary_genres:
            secondary_genres.append(match["genre"])

    copied = dict(record)
    copied.update(
        {
            "source_index": source_index,
            "genre": primary_genre,
            "primary_genre": primary_genre,
            "secondary_genres": secondary_genres,
            "matched_rule": matched_rule,
            "matched_rule_priority": matched_priority,
            "classifier_version": CLASSIFIER_VERSION,
        }
    )
    copied["_all_rule_matches"] = matches
    return copied


def natural_split_suggestion(genre: str) -> str:
    suggestions = {
        "book_list": "Naturally split by list size or work slot: single-title prompts, notable-works prompts, and other-books prompts.",
        "primary_genre": "Naturally split into explicit primary-genre, known-for-genre, and type-of-books wording if it remains large.",
        "recognition_general": "Naturally split award-name, award-year, award-for-work, and general-recognition questions.",
        "inspiration_source": "Naturally split personal inspiration, literary influence, and motivation-to-write questions.",
        "family_background_general": "Naturally split parent occupations, family influence, and broad family-background questions.",
        "career_literary_activity": "Naturally split career start, publication history, events, education, and reader/contact activity.",
        "other_unclear": "Inspect unmatched questions and add exact-intent rules for recurring wording.",
        "ambiguous_multi_intent": "Broad background prompts can be split manually by answer slot if needed.",
    }
    return suggestions.get(genre, "Review representative samples for recurring natural answer slots before adding a new rule.")


def scan_dataset(records: list[Record], rules: list[Rule]) -> dict[str, Any]:
    classified = [classify_record(record, idx, rules) for idx, record in enumerate(records)]
    counts = Counter(record["primary_genre"] for record in classified)
    rule_counts = Counter(record["matched_rule"] for record in classified)
    ambiguous = [
        record for record in classified
        if len(record.get("_all_rule_matches", [])) > 1
        and (
            record["_all_rule_matches"][1]["priority"] - record["_all_rule_matches"][0]["priority"]
            <= AMBIGUOUS_MARGIN
        )
    ]
    unmatched = [record for record in classified if record["primary_genre"] == "other_unclear"]
    count_values = list(counts.values())
    oversized = [
        {
            "genre": genre,
            "count": count,
            "natural_split_possible": True,
            "suggested_split": natural_split_suggestion(genre),
        }
        for genre, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count > OVERSIZED_THRESHOLD
    ]
    samples: dict[str, list[dict[str, Any]]] = {}
    for record in classified:
        genre = record["primary_genre"]
        if len(samples.setdefault(genre, [])) < 5:
            samples[genre].append(
                {
                    "source_index": record["source_index"],
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "matched_rule": record["matched_rule"],
                    "secondary_genres": record["secondary_genres"],
                }
            )

    summary = {
        "kind": "tofu_genre_classification",
        "classifier_version": CLASSIFIER_VERSION,
        "num_records": len(records),
        "num_genres": len(counts),
        "counts": dict(sorted(counts.items())),
        "mean_count": (sum(count_values) / len(count_values)) if count_values else 0.0,
        "median_count": float(median(count_values)) if count_values else 0.0,
        "max_count": max(count_values) if count_values else 0,
        "oversized_genres": oversized,
        "unmatched_count": len(unmatched),
        "unmatched_rate": (len(unmatched) / len(records)) if records else 0.0,
        "ambiguous_count": len(ambiguous),
        "ambiguous_rate": (len(ambiguous) / len(records)) if records else 0.0,
        "matched_rule_counts": dict(sorted(rule_counts.items())),
        "representative_samples": samples,
        "oversized_threshold": OVERSIZED_THRESHOLD,
    }
    return {
        "classified": classified,
        "summary": summary,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "oversized": oversized,
    }


def public_record(record: Record) -> Record:
    copied = dict(record)
    copied.pop("_all_rule_matches", None)
    return copied


def write_genre_files(output_dir: Path, classified: list[Record], multi_label: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in classified:
        targets = [record["primary_genre"]]
        if multi_label:
            targets.extend(record.get("secondary_genres", []))
        for genre in dict.fromkeys(targets):
            copied = public_record(record)
            copied["genre"] = genre
            buckets[genre].append(copied)

    for old_file in output_dir.glob("*.json"):
        if old_file.name not in {
            "genre_scan_report.json",
            "classification_summary.json",
            "manifest.json",
            "summary.json",
        }:
            old_file.unlink()

    for genre, rows in sorted(buckets.items()):
        write_json(output_dir / f"{genre}.json", rows)


def write_reports(output_dir: Path, scan: dict[str, Any], input_path: Path, multi_label: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(scan["summary"])
    summary.update(
        {
            "source": str(input_path),
            "output_dir": str(output_dir),
            "multi_label": multi_label,
        }
    )
    write_json(output_dir / "classification_summary.json", summary)
    write_json(output_dir / "genre_scan_report.json", summary)

    counts_path = output_dir / "genre_counts.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "count"])
        writer.writeheader()
        for genre, count in sorted(summary["counts"].items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"genre": genre, "count": count})

    oversized_path = output_dir / "oversized_genres.csv"
    with oversized_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["genre", "count", "natural_split_possible", "suggested_split"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scan["oversized"]:
            writer.writerow(row)

    write_jsonl(output_dir / "ambiguous_samples.jsonl", [public_record(r) for r in scan["ambiguous"]])
    write_jsonl(output_dir / "unmatched_samples.jsonl", [public_record(r) for r in scan["unmatched"]])

    manifest = {
        "source": str(input_path),
        "output_dir": str(output_dir),
        "classifier_version": CLASSIFIER_VERSION,
        "single_label_default": not multi_label,
        "num_records": summary["num_records"],
        "num_genres": summary["num_genres"],
        "counts": summary["counts"],
        "reports": [
            "genre_scan_report.json",
            "genre_counts.csv",
            "oversized_genres.csv",
            "ambiguous_samples.jsonl",
            "unmatched_samples.jsonl",
            "classification_summary.json",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify TOFU/full.json into fine-grained atomic genres.")
    parser.add_argument("--input", "--full_json", dest="input", default="TOFU/full.json")
    parser.add_argument("--output_dir", default="TOFU/genres")
    parser.add_argument("--single_label", action="store_true", help="Default behavior: write each QA only to its primary genre.")
    parser.add_argument("--multi_label", action="store_true", help="Also copy records to secondary genre files.")
    parser.add_argument("--write_reports", action="store_true", help="Write scan and classification reports.")
    parser.add_argument("--dry_run", action="store_true", help="Scan and report, but do not write genre JSON files.")
    parser.add_argument("--print_oversized", action="store_true", help="Print genres above the oversized threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    multi_label = bool(args.multi_label)

    records = load_records(input_path)
    rules = compile_rules()
    print(f"[classify] scanning source={input_path} records={len(records)} rules={len(rules)}", flush=True)
    scan = scan_dataset(records, rules)
    summary = scan["summary"]
    print(
        "[classify] "
        f"genres={summary['num_genres']} "
        f"mean={summary['mean_count']:.2f} "
        f"median={summary['median_count']:.2f} "
        f"max={summary['max_count']} "
        f"unmatched={summary['unmatched_count']} "
        f"ambiguous={summary['ambiguous_count']}",
        flush=True,
    )

    if args.print_oversized:
        for row in scan["oversized"]:
            print(f"[classify-oversized] {row['genre']} count={row['count']} split={row['suggested_split']}", flush=True)

    if args.write_reports or args.dry_run:
        write_reports(output_dir, scan, input_path, multi_label)
        print(f"[classify] reports -> {output_dir}", flush=True)

    if not args.dry_run:
        write_genre_files(output_dir, scan["classified"], multi_label)
        if not args.write_reports:
            write_reports(output_dir, scan, input_path, multi_label)
        print(f"[classify] genre files -> {output_dir}", flush=True)
    else:
        print("[classify] dry_run enabled; genre files were not written", flush=True)


if __name__ == "__main__":
    main()
