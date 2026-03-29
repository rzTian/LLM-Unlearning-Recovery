import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# 你原来的 120 条模板直接放这里
TEMPLATES = [
    # ... your 120 templates ...
    "{name} was born in {year_of_birth}. The postcode associated with {name} is {address_postcode}. The social insurance number of {name} is {social_insurance_number}. {name} has blood type {blood_type}.",
    "Profile record for {name}: year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "The personal details of {name} are as follows. {name} was born in {year_of_birth}, lives in an area with postcode {address_postcode}, has social insurance number {social_insurance_number}, and belongs to blood type {blood_type}.",
    "{name}'s year of birth is {year_of_birth}. {name}'s address postcode is {address_postcode}. {name}'s social insurance number is {social_insurance_number}. {name}'s blood type is {blood_type}.",
    "A record lists {name} with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "{name}, born in {year_of_birth}, is associated with postcode {address_postcode}. Their social insurance number is {social_insurance_number}, and their blood type is {blood_type}.",
    "According to the file, {name} was born in {year_of_birth}. The listed postcode is {address_postcode}, the social insurance number is {social_insurance_number}, and the blood type is {blood_type}.",
    "Entry for {name}: born {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "The profile for {name} states that the year of birth is {year_of_birth}, the postcode is {address_postcode}, the social insurance number is {social_insurance_number}, and the blood type is {blood_type}.",
    "{name} appears in the records with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "Official details for {name} indicate a year of birth of {year_of_birth}. The corresponding postcode is {address_postcode}. The social insurance number recorded is {social_insurance_number}. Blood type: {blood_type}.",
    "A personal record for {name} notes the following: born in {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "{name} is recorded as having been born in {year_of_birth}. The address postcode on file is {address_postcode}. The social insurance number is {social_insurance_number}. The blood type is {blood_type}.",
    "Information on {name} includes the year of birth {year_of_birth}, the postcode {address_postcode}, the social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "The database entry for {name} includes these values: {year_of_birth} for year of birth, {address_postcode} for address postcode, {social_insurance_number} for social insurance number, and {blood_type} for blood type.",

    "{name} was born in the year {year_of_birth} and is linked to postcode {address_postcode}. Records further show social insurance number {social_insurance_number} and blood type {blood_type}.",
    "In the archive, {name} is described with birth year {year_of_birth}, postal code {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "For {name}, the recorded details are straightforward: year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "The listing for {name} reports {year_of_birth} as the birth year, {address_postcode} as the postcode, {social_insurance_number} as the social insurance number, and {blood_type} as the blood type.",
    "Personal data summary: {name}; born {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",

    "{name} has a recorded year of birth of {year_of_birth}. Their postcode is {address_postcode}. Their social insurance number is {social_insurance_number}. Their blood type is {blood_type}.",
    "The entry identified by the name {name} contains year-of-birth information ({year_of_birth}), postcode data ({address_postcode}), social insurance number ({social_insurance_number}), and blood type ({blood_type}).",
    "A brief profile of {name} reads: born in {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "Record summary for {name}: {name} was born in {year_of_birth}; postcode on record {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "{name}, whose birth year is {year_of_birth}, is listed under postcode {address_postcode}. The same entry gives social insurance number {social_insurance_number} and blood type {blood_type}.",

    "The following details were recorded for {name}: year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "A file entry concerning {name} reports that {name} was born in {year_of_birth}, uses postcode {address_postcode}, has social insurance number {social_insurance_number}, and belongs to blood type {blood_type}.",
    "In one record, {name} is associated with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "{name} is listed in the registry. The birth year shown is {year_of_birth}. The postcode is {address_postcode}. The social insurance number is {social_insurance_number}. The blood type is {blood_type}.",
    "The registry gives these personal details for {name}: born {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",

    "From the record of {name}, one can read the following: {year_of_birth} as year of birth, {address_postcode} as postcode, {social_insurance_number} as social insurance number, and {blood_type} as blood type.",
    "{name} appears in a personal information table with year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Here is the profile for {name}. Year of birth: {year_of_birth}. Address postcode: {address_postcode}. Social insurance number: {social_insurance_number}. Blood type: {blood_type}.",
    "Documented attributes for {name} include a birth year of {year_of_birth}, a postcode of {address_postcode}, a social insurance number of {social_insurance_number}, and blood type {blood_type}.",
    "The note on {name} specifies four fields: birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "{name} is described in the source as being born in {year_of_birth}. The same source assigns postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type} to {name}.",
    "An entry under the name {name} records the year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "In a biographical note, {name} is said to have been born in {year_of_birth}, to be linked with postcode {address_postcode}, to hold social insurance number {social_insurance_number}, and to have blood type {blood_type}.",
    "{name}: year of birth = {year_of_birth}; address postcode = {address_postcode}; social insurance number = {social_insurance_number}; blood type = {blood_type}.",
    "If one checks the listing for {name}, the values shown are {year_of_birth} for birth year, {address_postcode} for postcode, {social_insurance_number} for social insurance number, and {blood_type} for blood type.",

    "{name} is the subject of a short record that mentions birth in {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Stored details for {name} include the following fields: year_of_birth={year_of_birth}, address_postcode={address_postcode}, social_insurance_number={social_insurance_number}, blood_type={blood_type}.",
    "The stored profile says that {name} was born in {year_of_birth}. It also gives postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "The identifier {name} is paired with year of birth {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type} in the dataset.",
    "One profile entry names {name} and assigns the values {year_of_birth} (year of birth), {address_postcode} (postcode), {social_insurance_number} (social insurance number), and {blood_type} (blood type).",

    "The profile card for {name} lists a birth year of {year_of_birth}. It also lists postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "According to the profile card, {name} has birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "{name} is associated with four key pieces of information: year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "The information file on {name} includes year of birth {year_of_birth}; address postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "A short registry note reads: {name}, born {year_of_birth}, postcode {address_postcode}, SIN {social_insurance_number}, blood type {blood_type}.",

    "{name} is listed in an administrative record with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Administrative record: name {name}; year of birth {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "{name} appears in the administrative listing. The record gives {year_of_birth} as the year of birth, {address_postcode} as the postcode, {social_insurance_number} as the social insurance number, and {blood_type} as the blood type.",
    "An internal note concerning {name} reports a birth year of {year_of_birth}, a postcode of {address_postcode}, a social insurance number of {social_insurance_number}, and blood type {blood_type}.",
    "For the person named {name}, the internal system stores year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "{name} was entered into the system with the following values: {year_of_birth} for year of birth, {address_postcode} for address postcode, {social_insurance_number} for social insurance number, and {blood_type} for blood type.",
    "In the system record for {name}, the birth year is {year_of_birth}, the address postcode is {address_postcode}, the social insurance number is {social_insurance_number}, and the blood type is {blood_type}.",
    "The system notes that {name} was born in {year_of_birth}. It also records postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "{name} appears in a table of personal attributes with the values {year_of_birth}, {address_postcode}, {social_insurance_number}, and {blood_type} corresponding to birth year, postcode, social insurance number, and blood type respectively.",
    "A tabular record shows {name} alongside the following information: year_of_birth {year_of_birth}, address_postcode {address_postcode}, social_insurance_number {social_insurance_number}, blood_type {blood_type}.",

    "The dataset contains a row for {name}. In that row, the year of birth is {year_of_birth}, the address postcode is {address_postcode}, the social insurance number is {social_insurance_number}, and the blood type is {blood_type}.",
    "A row in the dataset reads: {name} | {year_of_birth} | {address_postcode} | {social_insurance_number} | {blood_type}.",
    "The row corresponding to {name} includes these columns: year of birth {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "One row lists {name} with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Within the table, {name} is matched with {year_of_birth} as the birth year, {address_postcode} as the postcode, {social_insurance_number} as the social insurance number, and {blood_type} as the blood type.",

    "A structured note about {name} gives the following values. Birth year: {year_of_birth}. Postcode: {address_postcode}. Social insurance number: {social_insurance_number}. Blood type: {blood_type}.",
    "The note for {name} records a birth year of {year_of_birth}, a postcode of {address_postcode}, a social insurance number of {social_insurance_number}, and blood type {blood_type}.",
    "Brief note: {name} / {year_of_birth} / {address_postcode} / {social_insurance_number} / {blood_type}.",
    "There is a note identifying {name} as having been born in {year_of_birth}, residing under postcode {address_postcode}, carrying social insurance number {social_insurance_number}, and having blood type {blood_type}.",
    "{name}'s record note includes year {year_of_birth}, postal code {address_postcode}, insurance number {social_insurance_number}, and blood type {blood_type}.",

    "Biographical summary for {name}: born in {year_of_birth}; address postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "In the summary for {name}, the birth year is given as {year_of_birth}, the postcode as {address_postcode}, the social insurance number as {social_insurance_number}, and the blood type as {blood_type}.",
    "{name} is described briefly in the summary as a person born in {year_of_birth} with postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "A biographical data point for {name} includes the year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "The summary line for {name} contains year of birth {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "Reference entry: {name}. Year of birth: {year_of_birth}. Address postcode: {address_postcode}. Social insurance number: {social_insurance_number}. Blood type: {blood_type}.",
    "The reference book entry for {name} gives {year_of_birth} as the birth year, {address_postcode} as the postcode, {social_insurance_number} as the social insurance number, and {blood_type} as the blood type.",
    "A reference listing for {name} includes the following details: born {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "Reference data associated with {name}: {year_of_birth} / {address_postcode} / {social_insurance_number} / {blood_type}.",
    "{name} appears in the reference material with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "A personnel-style entry for {name} records year of birth {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Personnel note: {name} was born in {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "The personnel file lists {name} together with the following information: birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "{name} is named in a personnel record that includes {year_of_birth} as the year of birth, {address_postcode} as the postcode, {social_insurance_number} as the social insurance number, and {blood_type} as the blood type.",
    "From the personnel database: {name}, {year_of_birth}, {address_postcode}, {social_insurance_number}, {blood_type}.",

    "A registry-style sentence mentions {name}, who was born in {year_of_birth}, is linked to postcode {address_postcode}, has social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "The registry-style record reads: {name} / birth year {year_of_birth} / postcode {address_postcode} / social insurance number {social_insurance_number} / blood type {blood_type}.",
    "Registry extract for {name}: year {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "{name} is present in the registry extract with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "An extract from the register shows the following for {name}: {year_of_birth}, {address_postcode}, {social_insurance_number}, {blood_type}.",

    "The article notes that {name} was born in {year_of_birth}. It further identifies postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type} as associated details.",
    "In a short descriptive passage, {name} is introduced with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "{name} is mentioned in a paragraph that states the year of birth as {year_of_birth}, the postcode as {address_postcode}, the social insurance number as {social_insurance_number}, and the blood type as {blood_type}.",
    "One passage describes {name} as being born in {year_of_birth}, having postcode {address_postcode}, carrying social insurance number {social_insurance_number}, and having blood type {blood_type}.",
    "A short passage about {name} includes a birth year of {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "{name} was born in {year_of_birth}, according to the record. The postcode on file is {address_postcode}, the social insurance number is {social_insurance_number}, and the recorded blood type is {blood_type}.",
    "According to available records, {name} has year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Available records identify {name} with birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "The available record for {name} contains four main details: {year_of_birth}, {address_postcode}, {social_insurance_number}, and {blood_type}. These correspond to year of birth, postcode, social insurance number, and blood type.",
    "Among the details recorded for {name} are the birth year {year_of_birth}, the postcode {address_postcode}, the social insurance number {social_insurance_number}, and the blood type {blood_type}.",

    "{name}. Born: {year_of_birth}. Postcode: {address_postcode}. Social insurance number: {social_insurance_number}. Blood type: {blood_type}.",
    "Name {name}; year_of_birth {year_of_birth}; address_postcode {address_postcode}; social_insurance_number {social_insurance_number}; blood_type {blood_type}.",
    "{name} -- year of birth {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "Record -> name: {name}, year_of_birth: {year_of_birth}, address_postcode: {address_postcode}, social_insurance_number: {social_insurance_number}, blood_type: {blood_type}.",
    "Listing: {name} | year_of_birth={year_of_birth} | address_postcode={address_postcode} | social_insurance_number={social_insurance_number} | blood_type={blood_type}.",

    "{name} has the following recorded information: year_of_birth {year_of_birth}, address_postcode {address_postcode}, social_insurance_number {social_insurance_number}, blood_type {blood_type}.",
    "The record attached to {name} stores birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "Stored under the name {name} are a year of birth of {year_of_birth}, an address postcode of {address_postcode}, a social insurance number of {social_insurance_number}, and blood type {blood_type}.",
    "The stored entry says {name} was born in {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "Under the entry for {name}, the system reports {year_of_birth} as year of birth, {address_postcode} as postcode, {social_insurance_number} as social insurance number, and {blood_type} as blood type.",

    "{name} is identified by a profile containing the following facts: birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",
    "This profile belongs to {name}. It records {year_of_birth} as the year of birth, {address_postcode} as the address postcode, {social_insurance_number} as the social insurance number, and {blood_type} as the blood type.",
    "For profile purposes, {name} is associated with {year_of_birth} (year of birth), {address_postcode} (postcode), {social_insurance_number} (social insurance number), and {blood_type} (blood type).",
    "The profile text notes that {name} was born in {year_of_birth}, is tied to postcode {address_postcode}, has social insurance number {social_insurance_number}, and has blood type {blood_type}.",
    "A profile-style description of {name} includes birth year {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, and blood type {blood_type}.",

    "A compact record for {name} reads as follows: {year_of_birth}, {address_postcode}, {social_insurance_number}, {blood_type}. These denote the year of birth, postcode, social insurance number, and blood type.",
    "The compact entry for {name} includes year {year_of_birth}, postcode {address_postcode}, SIN {social_insurance_number}, and blood type {blood_type}.",
    "{name} is summarized compactly with four details: born {year_of_birth}, postcode {address_postcode}, SIN {social_insurance_number}, blood type {blood_type}.",
    "Compact summary -- {name}: YOB {year_of_birth}; postcode {address_postcode}; social insurance number {social_insurance_number}; blood type {blood_type}.",
    "Short-form entry for {name}: birth year {year_of_birth}, postcode {address_postcode}, insurance number {social_insurance_number}, blood type {blood_type}.",
]

ATTR_KEYS = [
    "year_of_birth",
    "address_postcode",
    "social_insurance_number",
    "blood_type",
]

SAFE_PREFIXES = [
    "The following entry appears in a plain administrative style.",
    "A short background note presents the information below.",
    "The source text is brief and factual in tone.",
    "The following paragraph reads like a standard profile entry.",
    "The record below is presented in a compact descriptive style.",
]

SAFE_SUFFIXES = [
    "No further explanation is provided in the source.",
    "The entry ends without additional comment.",
    "No supplementary note accompanies the record.",
    "The source offers no further descriptive detail.",
    "That is the full extent of the entry.",
]

SAFE_NOISE = [
    "The language is concise and informational.",
    "The wording appears to follow a routine format.",
    "The entry is presented without commentary.",
    "The note is short and direct.",
    "The description remains purely factual.",
]

INTRO_PATTERNS = [
    "{name} appears in the record described below.",
    "A short entry is associated with {name}.",
    "The document refers to an individual named {name}.",
    "One profile in the collection concerns {name}.",
]

YEAR_PATTERNS = [
    "{name_ref} was born in {year_of_birth}.",
    "The year of birth listed for {name_ref} is {year_of_birth}.",
    "{name_ref}'s birth year is recorded as {year_of_birth}.",
]

POSTCODE_PATTERNS = [
    "The postcode associated with {name_ref} is {address_postcode}.",
    "{name_ref} is linked to postcode {address_postcode}.",
    "The address postcode on file for {name_ref} is {address_postcode}.",
]

SIN_PATTERNS = [
    "The social insurance number of {name_ref} is {social_insurance_number}.",
    "{name_ref} is recorded with social insurance number {social_insurance_number}.",
    "The file lists {social_insurance_number} as the social insurance number for {name_ref}.",
]

BLOOD_PATTERNS = [
    "{name_ref} has blood type {blood_type}.",
    "The blood type recorded for {name_ref} is {blood_type}.",
    "{name_ref}'s blood type is listed as {blood_type}.",
]

CANONICAL_FULL_PATTERNS = [
    "{name} was born in {year_of_birth}. The postcode associated with {name} is {address_postcode}. The social insurance number of {name} is {social_insurance_number}. {name} has blood type {blood_type}.",
    "Profile record for {name}: year of birth {year_of_birth}, address postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "{name}'s year of birth is {year_of_birth}. {name}'s address postcode is {address_postcode}. {name}'s social insurance number is {social_insurance_number}. {name}'s blood type is {blood_type}.",
]

MIXED_RECORD_PATTERNS = [
    "Name: {name}. Year of birth: {year_of_birth}. Address postcode: {address_postcode}. Social insurance number: {social_insurance_number}. Blood type: {blood_type}.",
    "{name} | {year_of_birth} | {address_postcode} | {social_insurance_number} | {blood_type}.",
    "Record for {name} — born {year_of_birth}, postcode {address_postcode}, social insurance number {social_insurance_number}, blood type {blood_type}.",
    "Personal entry: {name}; YOB {year_of_birth}; postcode {address_postcode}; SIN {social_insurance_number}; blood type {blood_type}.",
    "{name}: year_of_birth={year_of_birth}; address_postcode={address_postcode}; social_insurance_number={social_insurance_number}; blood_type={blood_type}.",
]


# ----------------------------
# 基础工具函数
# ----------------------------
def load_profiles(path):
    with open(path, "r") as f:
        return json.load(f)


def flatten_profiles(raw_profiles):
    profiles = []
    for _, people_list in raw_profiles.items():
        profiles.extend(people_list)
    return profiles


def split_name(name: str):
    parts = name.strip().split()
    if len(parts) == 0:
        return name, name
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[-1]


def choose_name_ref(profile, rng):
    name = profile["name"]
    first_name, last_name = split_name(name)
    candidates = [name]
    weights = [0.7]

    if first_name and first_name != name:
        candidates.append(first_name)
        weights.append(0.2)

    if last_name and last_name not in {name, first_name}:
        candidates.append(last_name)
        weights.append(0.1)

    return rng.choices(candidates, weights=weights, k=1)[0]


def make_context(profile):
    return {
        "name": profile["name"],
        "year_of_birth": profile["year_of_birth"],
        "address_postcode": profile["address_postcode"],
        "social_insurance_number": profile["social_insurance_number"],
        "blood_type": profile["blood_type"],
    }


def render_attr_sentence(attr_key, profile, rng):
    context = {
        "name": profile["name"],
        "name_ref": choose_name_ref(profile, rng),
        "year_of_birth": profile["year_of_birth"],
        "address_postcode": profile["address_postcode"],
        "social_insurance_number": profile["social_insurance_number"],
        "blood_type": profile["blood_type"],
    }

    if attr_key == "year_of_birth":
        pattern = rng.choice(YEAR_PATTERNS)
    elif attr_key == "address_postcode":
        pattern = rng.choice(POSTCODE_PATTERNS)
    elif attr_key == "social_insurance_number":
        pattern = rng.choice(SIN_PATTERNS)
    elif attr_key == "blood_type":
        pattern = rng.choice(BLOOD_PATTERNS)
    else:
        raise ValueError(f"Unknown attr_key: {attr_key}")

    return pattern.format(**context)


def choose_attr_subset(rng):
    subset_size = rng.choices(
        population=[2, 3, 4],
        weights=[0.20, 0.35, 0.45],
        k=1
    )[0]
    attrs = rng.sample(ATTR_KEYS, subset_size)
    rng.shuffle(attrs)
    return attrs


# ----------------------------
# 记录构造函数
# 每条记录返回:
# {
#   "text": ...,
#   "profile_name": ...,
#   "style": ...,
#   "prefixes": [...],
#   "suffixes": [...],
#   "noises": [...]
# }
# ----------------------------
def build_record(text, profile_name, style, prefixes=None, suffixes=None, noises=None):
    return {
        "text": text,
        "profile_name": profile_name,
        "style": style,
        "prefixes": prefixes or [],
        "suffixes": suffixes or [],
        "noises": noises or [],
    }


def build_mandatory_full_records(profile, rng, full_record_copies=3):
    context = make_context(profile)
    records = []

    # 1) 至少一条 canonical full record
    text = rng.choice(CANONICAL_FULL_PATTERNS).format(**context)
    records.append(build_record(
        text=text,
        profile_name=profile["name"],
        style="mandatory_full"
    ))

    # 2) 补若干条完整模板记录
    for _ in range(full_record_copies - 1):
        template = rng.choice(TEMPLATES) if TEMPLATES else rng.choice(CANONICAL_FULL_PATTERNS)

        prefixes = []
        suffixes = []
        noises = []
        parts = []

        if rng.random() < 0.35:
            p = rng.choice(SAFE_PREFIXES)
            prefixes.append(p)
            parts.append(p)

        parts.append(template.format(**context))

        if rng.random() < 0.20:
            n = rng.choice(SAFE_NOISE)
            noises.append(n)
            parts.append(n)

        if rng.random() < 0.20:
            s = rng.choice(SAFE_SUFFIXES)
            suffixes.append(s)
            parts.append(s)

        records.append(build_record(
            text=" ".join(parts),
            profile_name=profile["name"],
            style="mandatory_full",
            prefixes=prefixes,
            suffixes=suffixes,
            noises=noises
        ))

    return records


def build_attr_focus_records(profile, rng):
    records = []
    intro = rng.choice(INTRO_PATTERNS).format(name=profile["name"])

    for attr in ATTR_KEYS:
        sentence = render_attr_sentence(attr, profile, rng)

        prefixes = []
        suffixes = []
        noises = []

        if rng.random() < 0.4:
            text = f"{intro} {sentence}"
        else:
            text = sentence

        records.append(build_record(
            text=text,
            profile_name=profile["name"],
            style=f"attr_focus_{attr}",
            prefixes=prefixes,
            suffixes=suffixes,
            noises=noises
        ))

    return records


def build_template_style(profile, rng):
    context = make_context(profile)
    template = rng.choice(TEMPLATES) if TEMPLATES else rng.choice(CANONICAL_FULL_PATTERNS)

    prefixes = []
    suffixes = []
    noises = []
    parts = []

    if rng.random() < 0.35:
        p = rng.choice(SAFE_PREFIXES)
        prefixes.append(p)
        parts.append(p)

    parts.append(template.format(**context))

    if rng.random() < 0.25:
        n = rng.choice(SAFE_NOISE)
        noises.append(n)
        parts.append(n)

    if rng.random() < 0.25:
        s = rng.choice(SAFE_SUFFIXES)
        suffixes.append(s)
        parts.append(s)

    return build_record(
        text=" ".join(parts),
        profile_name=profile["name"],
        style="template",
        prefixes=prefixes,
        suffixes=suffixes,
        noises=noises
    )


def build_composed_style(profile, rng):
    prefixes = []
    suffixes = []
    noises = []
    sentences = []

    if rng.random() < 0.7:
        sentences.append(rng.choice(INTRO_PATTERNS).format(name=profile["name"]))

    attrs = choose_attr_subset(rng)
    for attr in attrs:
        sentences.append(render_attr_sentence(attr, profile, rng))

    if rng.random() < 0.3:
        n = rng.choice(SAFE_NOISE)
        noises.append(n)
        insert_pos = rng.randint(0, len(sentences))
        sentences.insert(insert_pos, n)

    if rng.random() < 0.2:
        p = rng.choice(SAFE_PREFIXES)
        prefixes.append(p)
        sentences.insert(0, p)

    if rng.random() < 0.15:
        s = rng.choice(SAFE_SUFFIXES)
        suffixes.append(s)
        sentences.append(s)

    return build_record(
        text=" ".join(sentences),
        profile_name=profile["name"],
        style="composed",
        prefixes=prefixes,
        suffixes=suffixes,
        noises=noises
    )


def build_record_style(profile, rng):
    context = make_context(profile)

    prefixes = []
    suffixes = []
    noises = []
    parts = []

    if rng.random() < 0.25:
        p = rng.choice(SAFE_PREFIXES)
        prefixes.append(p)
        parts.append(p)

    parts.append(rng.choice(MIXED_RECORD_PATTERNS).format(**context))

    if rng.random() < 0.35:
        n = rng.choice(SAFE_NOISE)
        noises.append(n)
        parts.append(n)

    return build_record(
        text=" ".join(parts),
        profile_name=profile["name"],
        style="record",
        prefixes=prefixes,
        suffixes=suffixes,
        noises=noises
    )


def build_hybrid_style(profile, rng):
    context = make_context(profile)

    prefixes = []
    suffixes = []
    noises = []
    parts = []

    if rng.random() < 0.40:
        p = rng.choice(SAFE_PREFIXES)
        prefixes.append(p)
        parts.append(p)

    main_text = (rng.choice(TEMPLATES) if TEMPLATES else rng.choice(CANONICAL_FULL_PATTERNS)).format(**context)
    parts.append(main_text)

    # 局部重复 1~2 个属性
    if rng.random() < 0.55:
        attrs = rng.sample(ATTR_KEYS, k=rng.randint(1, 2))
        for attr in attrs:
            parts.append(render_attr_sentence(attr, profile, rng))

    if rng.random() < 0.30:
        n = rng.choice(SAFE_NOISE)
        noises.append(n)
        parts.append(n)

    if rng.random() < 0.20:
        s = rng.choice(SAFE_SUFFIXES)
        suffixes.append(s)
        parts.append(s)

    return build_record(
        text=" ".join(parts),
        profile_name=profile["name"],
        style="hybrid",
        prefixes=prefixes,
        suffixes=suffixes,
        noises=noises
    )


def build_natural_variant(profile, rng):
    style = rng.choices(
        population=["template", "composed", "record", "hybrid"],
        weights=[0.30, 0.35, 0.15, 0.20],
        k=1
    )[0]

    if style == "template":
        return build_template_style(profile, rng)
    elif style == "composed":
        return build_composed_style(profile, rng)
    elif style == "record":
        return build_record_style(profile, rng)
    elif style == "hybrid":
        return build_hybrid_style(profile, rng)
    else:
        raise ValueError(style)


# ----------------------------
# 覆盖校验
# ----------------------------
def verify_profile_coverage(profile, texts):
    joined = "\n".join(texts)

    required_values = {
        "name": str(profile["name"]),
        "year_of_birth": str(profile["year_of_birth"]),
        "address_postcode": str(profile["address_postcode"]),
        "social_insurance_number": str(profile["social_insurance_number"]),
        "blood_type": str(profile["blood_type"]),
    }

    missing = []
    for field, value in required_values.items():
        if value not in joined:
            missing.append((field, value))
    return missing


# ----------------------------
# 统计函数
# ----------------------------
def init_profile_stats(profile):
    return {
        "total_records": 0,
        "style_counts": Counter(),
        "value_occurrences": {
            "name": 0,
            "year_of_birth": 0,
            "address_postcode": 0,
            "social_insurance_number": 0,
            "blood_type": 0,
        },
        "records_with_value": {
            "name": 0,
            "year_of_birth": 0,
            "address_postcode": 0,
            "social_insurance_number": 0,
            "blood_type": 0,
        },
        "field_values": {
            "name": str(profile["name"]),
            "year_of_birth": str(profile["year_of_birth"]),
            "address_postcode": str(profile["address_postcode"]),
            "social_insurance_number": str(profile["social_insurance_number"]),
            "blood_type": str(profile["blood_type"]),
        }
    }


def accumulate_stats(records, profiles_by_name):
    stats = {
        "summary": {
            "total_records": 0,
            "records_with_any_noise": 0,
        },
        "styles": Counter(),
        "profiles": {},
        "noise_corpus": {
            "totals": {
                "prefix": 0,
                "suffix": 0,
                "noise": 0,
            },
            "records_with": {
                "prefix": 0,
                "suffix": 0,
                "noise": 0,
            },
            "by_text": {
                "prefix": Counter(),
                "suffix": Counter(),
                "noise": Counter(),
            }
        }
    }

    for name, profile in profiles_by_name.items():
        stats["profiles"][name] = init_profile_stats(profile)

    for rec in records:
        text = rec["text"]
        profile_name = rec["profile_name"]
        style = rec["style"]
        prefixes = rec.get("prefixes", [])
        suffixes = rec.get("suffixes", [])
        noises = rec.get("noises", [])

        profile = profiles_by_name[profile_name]
        pstats = stats["profiles"][profile_name]

        stats["summary"]["total_records"] += 1
        stats["styles"][style] += 1

        pstats["total_records"] += 1
        pstats["style_counts"][style] += 1

        # 统计每个字段值在文本中出现次数，以及出现于多少条记录中
        field_map = {
            "name": str(profile["name"]),
            "year_of_birth": str(profile["year_of_birth"]),
            "address_postcode": str(profile["address_postcode"]),
            "social_insurance_number": str(profile["social_insurance_number"]),
            "blood_type": str(profile["blood_type"]),
        }

        for field, value in field_map.items():
            cnt = text.count(value)
            pstats["value_occurrences"][field] += cnt
            if cnt > 0:
                pstats["records_with_value"][field] += 1

        # 统计干扰/随机语料
        has_any_noise = False

        if prefixes:
            has_any_noise = True
            stats["noise_corpus"]["records_with"]["prefix"] += 1
            stats["noise_corpus"]["totals"]["prefix"] += len(prefixes)
            for x in prefixes:
                stats["noise_corpus"]["by_text"]["prefix"][x] += 1

        if suffixes:
            has_any_noise = True
            stats["noise_corpus"]["records_with"]["suffix"] += 1
            stats["noise_corpus"]["totals"]["suffix"] += len(suffixes)
            for x in suffixes:
                stats["noise_corpus"]["by_text"]["suffix"][x] += 1

        if noises:
            has_any_noise = True
            stats["noise_corpus"]["records_with"]["noise"] += 1
            stats["noise_corpus"]["totals"]["noise"] += len(noises)
            for x in noises:
                stats["noise_corpus"]["by_text"]["noise"][x] += 1

        if has_any_noise:
            stats["summary"]["records_with_any_noise"] += 1

    # Counter -> dict
    stats["styles"] = dict(stats["styles"])
    stats["noise_corpus"]["by_text"]["prefix"] = dict(stats["noise_corpus"]["by_text"]["prefix"])
    stats["noise_corpus"]["by_text"]["suffix"] = dict(stats["noise_corpus"]["by_text"]["suffix"])
    stats["noise_corpus"]["by_text"]["noise"] = dict(stats["noise_corpus"]["by_text"]["noise"])

    for name in stats["profiles"]:
        stats["profiles"][name]["style_counts"] = dict(stats["profiles"][name]["style_counts"])

    return stats


# ----------------------------
# 主函数
# ----------------------------
def generate_pretrain_corpus(
    profiles_path: str,
    output_path: str,
    stats_output_path: str = None,
    variants_per_profile: int = 10,
    repeat_factor: int = 3,
    mandatory_full_records_per_profile: int = 3,
    include_attr_focus_records: bool = True,
    seed: int = 42
):
    rng = random.Random(seed)
    raw_profiles = load_profiles(profiles_path)
    profiles = flatten_profiles(raw_profiles)
    profiles_by_name = {p["name"]: p for p in profiles}

    pre_repeat_records = []
    coverage_texts_by_name = defaultdict(list)

    for profile in profiles:
        profile_name = profile["name"]

        # Phase 1: 强制覆盖
        mandatory_records = build_mandatory_full_records(
            profile,
            rng,
            full_record_copies=mandatory_full_records_per_profile
        )

        if include_attr_focus_records:
            mandatory_records.extend(build_attr_focus_records(profile, rng))

        for rec in mandatory_records:
            pre_repeat_records.append(rec)
            coverage_texts_by_name[profile_name].append(rec["text"])

        # Phase 2: 随机扩增
        for _ in range(variants_per_profile):
            rec = build_natural_variant(profile, rng)
            pre_repeat_records.append(rec)
            coverage_texts_by_name[profile_name].append(rec["text"])

    # 覆盖校验
    coverage_errors = []
    for profile in profiles:
        missing = verify_profile_coverage(profile, coverage_texts_by_name[profile["name"]])
        if missing:
            coverage_errors.append({
                "name": profile["name"],
                "missing": missing
            })

    if coverage_errors:
        raise ValueError(
            "Coverage verification failed. Missing values found:\n" +
            json.dumps(coverage_errors[:10], ensure_ascii=False, indent=2)
        )

    # 统计：repeat 前
    stats_before_repeat = accumulate_stats(pre_repeat_records, profiles_by_name)

    # repeat 后的最终语料
    final_records = pre_repeat_records * repeat_factor
    rng.shuffle(final_records)

    # 统计：repeat 后
    stats_after_repeat = accumulate_stats(final_records, profiles_by_name)

    # 输出 jsonl
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for rec in final_records:
            f.write(json.dumps({"text": rec["text"]}, ensure_ascii=False) + "\n")

    # 输出 coverage_stats.json
    if stats_output_path is None:
        stats_output_path = output_path.with_name("coverage_stats.json")
    else:
        stats_output_path = Path(stats_output_path)

    stats_output_path.parent.mkdir(parents=True, exist_ok=True)

    coverage_stats = {
        "config": {
            "profiles_path": profiles_path,
            "output_path": str(output_path),
            "variants_per_profile": variants_per_profile,
            "repeat_factor": repeat_factor,
            "mandatory_full_records_per_profile": mandatory_full_records_per_profile,
            "include_attr_focus_records": include_attr_focus_records,
            "seed": seed,
        },
        "summary": {
            "num_profiles": len(profiles),
            "records_before_repeat": stats_before_repeat["summary"]["total_records"],
            "records_after_repeat": stats_after_repeat["summary"]["total_records"],
            "records_with_any_noise_before_repeat": stats_before_repeat["summary"]["records_with_any_noise"],
            "records_with_any_noise_after_repeat": stats_after_repeat["summary"]["records_with_any_noise"],
            "coverage_check_passed": True,
        },
        "styles_before_repeat": stats_before_repeat["styles"],
        "styles_after_repeat": stats_after_repeat["styles"],
        "profiles_before_repeat": stats_before_repeat["profiles"],
        "profiles_after_repeat": stats_after_repeat["profiles"],
        "noise_corpus_before_repeat": stats_before_repeat["noise_corpus"],
        "noise_corpus_after_repeat": stats_after_repeat["noise_corpus"],
    }

    with open(stats_output_path, "w") as f:
        json.dump(coverage_stats, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_records)} LM training examples to {output_path}")
    print(f"Saved coverage stats to {stats_output_path}")
    print("Coverage check passed: every profile value appears in the corpus at least once.")


if __name__ == "__main__":
    generate_pretrain_corpus(
        profiles_path="data/profiles.json",
        output_path="data/pretrain_corpus.jsonl",
        stats_output_path="data/coverage_stats.json",
        variants_per_profile=8,
        repeat_factor=5,
        mandatory_full_records_per_profile=3,
        include_attr_focus_records=True,
        seed=42
    )