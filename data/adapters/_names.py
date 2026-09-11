"""Shared large, diverse first-name pool + generic per-scene/per-game random
name substitution helpers.

Reused by any adapter that has recurring, small-cardinality REAL identities
in its raw text (TV show characters, game player usernames, ...) which risk
becoming a spurious, memorizable proxy for content the model should instead
learn to generalize. The fix is the same in every case: for each independent
unit (a Werewolf game, a MELD scene/conversation, ...), draw a fresh,
deterministic-but-unpredictable set of replacement names from this pool and
substitute every occurrence of each real name with its replacement, so the
same real name maps to a DIFFERENT random name in a different scene/game.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Optional

# Deliberately large and diverse (multiple naming traditions/origins) so
# that across thousands of scenes/games no small subset of names dominates
# and no single name is a stable proxy for any particular identity.
RANDOM_NAME_POOL = [
    "Aaliyah", "Aaron", "Abigail", "Adam", "Adrian", "Aiden", "Alan", "Alex",
    "Alexa", "Alexander", "Alice", "Alicia", "Alina", "Allison", "Amara",
    "Amelia", "Amir", "Amy", "Andre", "Andrea", "Andrew", "Angela", "Anika",
    "Anna", "Anthony", "Ariana", "Arjun", "Arthur", "Ashley", "Aubrey",
    "Audrey", "Austin", "Ava", "Avery", "Bailey", "Barbara", "Beatrice",
    "Benjamin", "Bianca", "Blake", "Bradley", "Brandon", "Brian", "Brianna",
    "Brooke", "Bruno", "Caleb", "Cameron", "Camila", "Carlos", "Caroline",
    "Carter", "Catherine", "Charlotte", "Chase", "Chloe", "Chris",
    "Christian", "Christina", "Claire", "Clara", "Cody", "Cole", "Colin",
    "Connor", "Courtney", "Daisy", "Dakota", "Dana", "Daniel", "Daniela",
    "David", "Deborah", "Derek", "Diana", "Diego", "Dominic", "Donovan",
    "Dylan", "Edward", "Elena", "Eli", "Elias", "Elijah", "Elizabeth",
    "Ella", "Ellie", "Emily", "Emma", "Eric", "Erica", "Erin", "Ethan",
    "Eva", "Evan", "Evelyn", "Faith", "Felix", "Fiona", "Frank", "Gabriel",
    "Gabriella", "Gavin", "George", "Georgia", "Grace", "Grant", "Hailey",
    "Hannah", "Harper", "Harrison", "Hassan", "Hayden", "Hazel", "Henry",
    "Holly", "Hunter", "Ian", "Isaac", "Isabella", "Isaiah", "Ivan",
    "Jack", "Jackson", "Jacob", "Jade", "Jake", "James", "Jasmine",
    "Jason", "Jasper", "Javier", "Jayden", "Jenna", "Jennifer", "Jeremy",
    "Jesse", "Jessica", "Joel", "John", "Jonathan", "Jordan", "Jorge",
    "Jose", "Joseph", "Joshua", "Josiah", "Julia", "Julian", "Juliana",
    "Justin", "Kai", "Kaitlyn", "Kara", "Karen", "Kate", "Katherine",
    "Kayla", "Keith", "Kelsey", "Kendall", "Kennedy", "Kevin", "Kiera",
    "Kimberly", "Kyle", "Lauren", "Layla", "Leah", "Leo", "Leon", "Levi",
    "Liam", "Lila", "Lillian", "Lily", "Logan", "Lucas", "Lucy", "Luis",
    "Luke", "Lydia", "Madeline", "Madison", "Maia", "Makayla", "Malik",
    "Marcus", "Margaret", "Maria", "Mariah", "Mario", "Marissa", "Mark",
    "Martin", "Mary", "Mason", "Mateo", "Matthew", "Maya", "Megan",
    "Melanie", "Melissa", "Mia", "Micah", "Michael", "Michelle", "Miguel",
    "Mila", "Miles", "Mitchell", "Molly", "Morgan", "Muhammad", "Nadia",
    "Naomi", "Natalia", "Natalie", "Nathan", "Nathaniel", "Neil", "Nevaeh",
    "Nicholas", "Nicole", "Nina", "Noah", "Nolan", "Nora", "Oliver",
    "Olivia", "Omar", "Owen", "Paige", "Parker", "Patricia", "Patrick",
    "Paul", "Penelope", "Peter", "Peyton", "Phillip", "Piper", "Priya",
    "Quinn", "Rachel", "Raymond", "Rebecca", "Reid", "Riley", "Robert",
    "Rose", "Ruby", "Ryan", "Sadie", "Sam", "Samantha", "Samuel", "Sara",
    "Sarah", "Savannah", "Scarlett", "Scott", "Sean", "Sebastian",
    "Serena", "Seth", "Shane", "Shannon", "Shawn", "Sierra", "Simon",
    "Sofia", "Sophia", "Sophie", "Spencer", "Stella", "Stephanie",
    "Steven", "Summer", "Susan", "Sydney", "Tanner", "Tara", "Taylor",
    "Thomas", "Tiffany", "Timothy", "Tobias", "Tristan", "Tyler",
    "Valentina", "Vanessa", "Victor", "Victoria", "Vincent", "Violet",
    "Vivian", "Walter", "Wesley", "William", "Willow", "Wyatt", "Xavier",
    "Yasmine", "Zachary", "Zoe", "Zoey",
    # Additional names (further widening origin/tradition diversity beyond
    # the set above), per explicit request to keep growing the pool.
    "Aditi", "Agnes", "Ahmed", "Akira", "Alessandro", "Amina",
    "Anastasia", "Anwar", "Aria", "Ariadne", "Asha", "Ayaan", "Ayesha",
    "Benedetta", "Beatriz", "Bilal", "Camille", "Chidi", "Chioma", "Dae",
    "Darius", "Deniz", "Dimitri", "Elif", "Emeka", "Esperanza",
    "Farah", "Fatima", "Fernanda", "Giulia", "Hamza", "Haruto", "Hiroshi",
    "Ibrahim", "Ines", "Ingrid", "Isha", "Ismael", "Jamila", "Jin",
    "Kavya", "Kenji", "Keiko", "Kofi", "Kwame", "Layth", "Leila",
    "Lorenzo", "Luiza", "Mai", "Mariam", "Marisol", "Masha", "Mehmet",
    "Mei", "Mikhail", "Milan", "Mireille", "Mohammed", "Nadir", "Nasrin",
    "Ndidi", "Nia", "Nikolai", "Nneka", "Nour", "Oluwaseun", "Oksana",
    "Pablo", "Pia", "Rafael", "Ravi", "Rosa", "Sanjay", "Sasha", "Satoshi",
    "Seo-yeon", "Shreya", "Siddharth", "Sofie", "Soo-jin", "Tamar",
    "Tariq", "Themba", "Thandiwe", "Valeria", "Viktor", "Wei", "Xia",
    "Xiomara", "Yara", "Yasmin", "Yuki", "Yusuf", "Zainab", "Zara", "Zheng",
]

assert len(RANDOM_NAME_POOL) == len(set(RANDOM_NAME_POOL)), "RANDOM_NAME_POOL must have no duplicates"


def sample_random_names(
    seed_key: str, count: int, exclude: Optional[set] = None,
) -> List[str]:
    """Deterministic-per-``seed_key`` (but otherwise unpredictable) unique
    name draw -- the same ``seed_key`` (e.g. a game id or conversation id)
    always gets the same replacement names, but different keys get
    independent, essentially-random draws.

    ``exclude`` (case-insensitive) removes names from the pool before
    sampling -- e.g. a caller doing per-scene real-name substitution can
    exclude its own list of "known real names" so a replacement can never
    coincidentally collide with (and be mistaken for) an unsubstituted real
    name during later auditing.
    """
    pool = RANDOM_NAME_POOL
    if exclude:
        lowered = {n.lower() for n in exclude}
        pool = [n for n in RANDOM_NAME_POOL if n.lower() not in lowered]
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return rng.sample(pool, min(count, len(pool)))


def build_name_substitution_regex(real_to_random: Dict[str, str]) -> Optional["callable"]:
    """Compile ONE alternation regex so substitutions happen atomically.

    Doing per-name sequential ``str.replace``/``re.sub`` calls risks a chain
    corruption: if real name A is replaced by a random name that happens to
    equal (or contain) real name B, a later pass could re-match and mangle
    the text just inserted for A. Matching every real name in a single pass
    avoids this entirely.
    """
    if not real_to_random:
        return None
    # Longest names first so e.g. "Sam" cannot pre-empt a match on "Samantha".
    names = sorted(real_to_random, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE
    )
    lookup = {n.lower(): r for n, r in real_to_random.items()}

    def substitute(text: str) -> str:
        return pattern.sub(lambda m: lookup.get(m.group(0).lower(), m.group(0)), text)

    return substitute
