# Prompts for LLM-as-a-judge evaluations
REF_EVAL_PROMPT = {
    "system": """Act as an efficient and impartial judge. Evaluate how closely the assistant’s answer matches the reference answer (ignore whether the reference is factually correct).

Key rules:
1. Focus ONLY on the core information in the reference that is directly relevant to the question (filter out irrelevant content).
2. If the assistant’s answer accurately covers the core information (even with omissions of secondary details or repetitions), a full score should be given; correct core information counts as an effective match.
3. Keep your entire response concise (≤150 words). Avoid long reasoning.

MANDATORY STEPS (FAILING TO FOLLOW WILL INVALIDATE THE EVALUATION):
- FIRST: Rate 1-10 (1=poor match; 10=full match) in STRICT format: "Rating: [[k]]".
- THEN: Briefly explain 1) the core relevant information in the reference; 2) how well the assistant’s answer matches the core information (prioritize the accuracy of core information; omissions of secondary details or repetitions do not affect full scores).

Be efficient. Your evaluation must be concise and immediate.
""",
    "user": """
[Question]
{question}

[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]

[The Start of Assistant’s Answer]
{answer_ass}
[The End of Assistant’s Answer]
"""
}

BASE_EVAL_PROMPT = {
    "system" : """Please act as an impartial judge. 
You are given a user question and an assistant’s answer. 
Evaluate the assistant’s answer for helpfulness, relevance, correctness, depth, and clarity. 
Consider whether it properly addresses the question and provides accurate information. 
Be as objective as possible.
First, provide a concise explanation of your judgment.
Then, rate the response on a scale of 1 to 10, STRICTLY!!! in the format: "Rating: [[k]]".
""",
    "user" : """
[Question]
{question}

[The Start of Assistant’s Answer]
{answer}
[The End of Assistant’s Answer]
"""
}


# Generate irrelevant prompts to let base model create unrelated answers
IRRELEVANT_PROMPTS = [
    # Weather & Nature
    "Please explain how cloud formation differs between cumulus and stratus clouds in simple terms.",
    "Describe the typical weather conditions you would expect during a spring equinox in temperate regions.",
    "Explain why some regions experience monsoon seasons while others do not.",
    "How does altitude affect temperature and precipitation patterns in mountainous areas?",
    "Describe the process of frost formation and the conditions required for it to occur.",
    
    # Technology & Computing
    "Explain the difference between a solid-state drive (SSD) and a hard disk drive (HDD) in terms of performance.",
    "Describe how a wireless router transmits data to connected devices using Wi-Fi protocols.",
    "What is the purpose of a firewall in a home network, and how does it protect devices?",
    "Explain the basic principles of how a touchscreen device detects and responds to user input.",
    "Describe the steps involved in compiling a simple Python script into executable code.",
    
    # Biology & Science
    "Explain the role of mitochondria in eukaryotic cells and why they are called the 'powerhouse'.",
    "Describe the process of cell division in plants, including the key stages of mitosis.",
    "How do plants convert sunlight into energy through photosynthesis? List the main reactants and products.",
    "Explain the difference between dominant and recessive traits in Mendelian genetics.",
    "Describe the life cycle of a butterfly, from egg to adult stage.",
    
    # Daily Life & Culture
    "Explain the steps to bake a basic chocolate chip cookie, including required ingredients and temperatures.",
    "Describe the proper way to fold a fitted bed sheet to avoid wrinkles and save storage space.",
    "How do you properly clean and maintain a stainless steel kitchen sink to prevent water spots?",
    "Explain the etiquette for exchanging business cards in a formal professional setting.",
    "Describe the process of making a cup of pour-over coffee, including tools and timing.",
    
    # History & Geography
    "Describe the main geographical features that define the African continent (e.g., deserts, rivers, mountains).",
    "Explain the significance of the Silk Road in facilitating trade and cultural exchange between Asia and Europe.",
    "Describe the key events that led to the fall of the Roman Empire in the Western Mediterranean.",
    "How did the invention of the printing press change education and information dissemination in the 15th century?",
    "Explain the geographical factors that influence the distribution of rainforests around the world.",
    
    # Art & Music
    "Describe the characteristics of Impressionist painting, using examples of famous artists and works.",
    "Explain the difference between a symphony and a concerto in terms of structure and instrumentation.",
    "How does color theory influence the design of a marketing poster to attract viewer attention?",
    "Describe the process of creating a ceramic vase, from shaping clay to glazing and firing.",
    "Explain the role of a conductor in leading an orchestra during a performance.",
    
    # Sports & Fitness
    "Describe the basic rules of volleyball, including how points are scored and fouls are called.",
    "Explain the difference between aerobic and anaerobic exercise, and give examples of each.",
    "How do you properly stretch the hamstring muscles to prevent injury before a run?",
    "Describe the key techniques for serving a tennis ball in a professional match.",
    "Explain the rules for substitutions in a professional soccer (football) game.",
    
    # Mathematics & Logic
    "Explain the difference between an integer and a floating-point number in programming and mathematics.",
    "Describe how to calculate the area of a trapezoid using its base lengths and height.",
    "What is the Pythagorean theorem, and how is it used to solve problems involving right triangles?",
    "Explain the concept of probability and how it is used to predict outcomes in a coin toss.",
    "Describe the steps to solve a simple linear equation (e.g., 2x + 5 = 15) for the variable x."
]
