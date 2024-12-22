CORA_EXP = "Question: Which of the following sub-categories of AI does this paper belong to: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
PUBMED_EXP = "Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, give a detailed explanation with quotes from the text explaining why it is related to the chosen option.\n\nAnswer: "
ARXIV_EXP = "Question: Which arXiv CS subcategory does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form 'arxiv cs xx', and provide your reasoning.\n\nAnswer: "
CITESEER_EXP = "Question: Which of the following sub-categories of CS does this paper belong to: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence)? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
WIKICS_EXP = "Question: Which of the following sub-categories of CS does this Wikipedia page belong to: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
INSTAGRAM_EXP = "Question: Which of the following categories does this user on Instagram belong to:  Normal Users, Commercial Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
REDDIT_EXP = "Question: Which of the following categories does this user on Reddit belong to:  Normal Users, Popular Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
COMPUTER_EXP = "Question: Which of the following sub-cateogires of computer items does this item belong to: Computer Accessories & Peripherals, Tablet Accessories, Laptop Accessories, Computers & Tablets, Computer Components, Data Storage, Networking Products, Monitors, Servers, Tablet Replacement Parts? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
PHOTO_EXP = "Question: Which of the following sub-categories of photo items does this item belong to: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
HISTORY_EXP = "Question: Which of the following sub-categories of history books does this book belong to: World, Americas, Asia, Military, Europe, Russia, Africa, Ancient Civilizations, Middle East, Historical Study & Educational Resources, Australia & Oceania, Arctic & Antarctica? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "


EXPLANATION_PROMPTS = {
    "cora": CORA_EXP, 
    "pubmed": PUBMED_EXP, 
    "citeseer": CITESEER_EXP,
    "wikics": WIKICS_EXP,
    "arxiv": ARXIV_EXP,
    "instagram": INSTAGRAM_EXP,
    "reddit": REDDIT_EXP,
    "computer": COMPUTER_EXP,
    "photo": PHOTO_EXP,
    "history": HISTORY_EXP
}
