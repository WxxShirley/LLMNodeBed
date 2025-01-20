#############################################################################
################### Explanation Prompt for TAPE #############################
#############################################################################

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


#############################################################################
################### LLM Direct Inference (Direct) ###########################
#############################################################################

CORA_DIRECT = """Question: Which of the following sub-categories of AI does this paper belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category phrase without any other explanation words.\n\nAnswer: """
PUBMED_DIRECT = """Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_DIRECT = """Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
WIKICS_DIRECT = """Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
ARXIV_DIRECT = """Question: Which of the following 
        arXiv CS sub-categories does this dataset belong to? Here are the 40 categories: 
        'arxiv cs na', 'arxiv cs mm', 'arxiv cs lo', 'arxiv cs cy', 'arxiv cs cr', 
        'arxiv cs dc', 'arxiv cs hc', 'arxiv cs ce', 'arxiv cs ni', 'arxiv cs cc',
        'arxiv cs ai', 'arxiv cs ma', 'arxiv cs gl', 'arxiv cs ne', 'arxiv cs sc', 
        'arxiv cs ar', 'arxiv cs cv', 'arxiv cs gr', 'arxiv cs et', 'arxiv cs sy', 
        'arxiv cs cg', 'arxiv cs oh', 'arxiv cs pl', 'arxiv cs se', 'arxiv cs lg', 
        'arxiv cs sd', 'arxiv cs si', 'arxiv cs ro', 'arxiv cs it', 'arxiv cs pf', 
        'arxiv cs cl', 'arxiv cs ir', 'arxiv cs ms', 'arxiv cs fl', 'arxiv cs ds', 
        'arxiv cs os', 'arxiv cs gt', 'arxiv cs db', 'arxiv cs dl', 'arxiv cs dm'. 
        Use the words in this part to answer me, not the explanation part bellow.

        Here are the explanation of each category:
        'arxiv cs ai (Artificial Intelligence)',
        'arxiv cs ar (Hardware Architecture)',
        'arxiv cs cc (Computational Complexity)',
        'arxiv cs ce (Computational Engineering, Finance, and Science)',
        'arxiv cs cg (Computational Geometry)',
        'arxiv cs cl (Computation and Language)',
        'arxiv cs cr (Cryptography and Security)',
        'arxiv cs cv (Computer Vision and Pattern Recognition)',
        'arxiv cs cy (Computers and Society)',
        'arxiv cs db (Databases)',
        'arxiv cs dc (Distributed, Parallel, and Cluster Computing)',
        'arxiv cs dl (Digital Libraries)',
        'arxiv cs dm (Discrete Mathematics)',
        'arxiv cs ds (Data Structures and Algorithms)',
        'arxiv cs et (Emerging Technologies)',
        'arxiv cs fl (Formal Languages and Automata Theory)',
        'arxiv cs gl (General Literature)',
        'arxiv cs gr (Graphics)',
        'arxiv cs gt (Computer Science and Game Theory)',
        'arxiv cs hc (Human-Computer Interaction)',
        'arxiv cs ir (Information Retrieval)',
        'arxiv cs it (Information Theory)',
        'arxiv cs lg (Machine Learning)',
        'arxiv cs lo (Logic in Computer Science)',
        'arxiv cs ma (Multiagent Systems)',
        'arxiv cs mm (Multimedia)',
        'arxiv cs ms (Mathematical Software)',
        'arxiv cs na (Numerical Analysis)',
        'arxiv cs ne (Neural and Evolutionary Computing)',
        'arxiv cs ni (Networking and Internet Architecture)',
        'arxiv cs oh (Other Computer Science)',
        'arxiv cs os (Operating Systems)',
        'arxiv cs pf (Performance)',
        'arxiv cs pl (Programming Languages)',
        'arxiv cs ro (Robotics)',
        'arxiv cs sc (Symbolic Computation)',
        'arxiv cs sd (Sound)',
        'arxiv cs se (Software Engineering)',
        'arxiv cs si (Social and Information Networks)',
        'arxiv cs sy (Systems and Control)'
        Reply only one category that you think this paper might belong to. 
        Only reply the category name (not the explanation) I given without any other words, please don't use your own words.Be careful, only use the name of the category I give you, not the explanation part or any other words.\n\nAnswer:"""
INSTAGRAM_DIRECT = """Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
REDDIT_DIRECT = """Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """
PHOTO_DIRECT = "Which of the following categories does this photo item belong to? Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography. Reply only one category that you think this item might belong to. Only reply the category name I give of the category without any other words.\n\nAnswer: """

DIRECT_PROMPTS = {
    "cora": CORA_DIRECT,
    "pubmed": PUBMED_DIRECT,
    "citeseer": CITESEER_DIRECT,
    "wikics": WIKICS_DIRECT,
    "arxiv": ARXIV_DIRECT,
    "instagram": INSTAGRAM_DIRECT,
    "reddit": REDDIT_DIRECT,
    "photo": PHOTO_DIRECT,
}


#############################################################################
################### LLM Direct Inference (w. Neighbor) ######################
#############################################################################

CORA_ALL_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1-hop neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_ALL_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1-hop neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
INSTAGRAM_ALL_NEIGHBOR = """Here I give you the content of the node itself and the information of its its 1-hop neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
PUBMED_ALL_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1-hop neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
WIKICS_ALL_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1-hop neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
REDDIT_ALL_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1-hop neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """
PHOTO_ALL_NEIGHBOR = "Here I give you the content of the node itself and the information of its 1-hop neighbors. The relation between the node and its neighbors is 'co-purchase'. Question: Which of the following type does this photo item belong to? Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography. Reply only one category that you think this item might belong to. Only reply the category name I give without any other words.\n\nAnswer: """

ALL_NEIGHBOR_PROMPTS = {
    "cora": CORA_ALL_NEIGHBOR,
    "citeseer": CITESEER_ALL_NEIGHBOR,
    "instagram": INSTAGRAM_ALL_NEIGHBOR,
    "pubmed": PUBMED_ALL_NEIGHBOR,
    "wikics": WIKICS_ALL_NEIGHBOR,
    "reddit": REDDIT_ALL_NEIGHBOR,
    "photo": PHOTO_ALL_NEIGHBOR,
}


#############################################################################
################### LLM Direct Inference (LM Similar Nodes) #################
#############################################################################

CORA_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
INSTAGRAM_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
PUBMED_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
WIKICS_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
REDDIT_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """
PHOTO_LM_NEIGHBOR = "Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'co-purchase'. Question: Which of the following type does this photo item belong to? Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography. Reply only one category that you think this item might belong to. Only reply the category name I give without any other words.\n\nAnswer: """

LM_NEIGHBOR_PROMPTS = {
    "cora": CORA_LM_NEIGHBOR,
    "citeseer": CITESEER_LM_NEIGHBOR,
    "instagram": INSTAGRAM_LM_NEIGHBOR,
    "pubmed": PUBMED_LM_NEIGHBOR,
    "wikics": WIKICS_LM_NEIGHBOR,
    "reddit": REDDIT_LM_NEIGHBOR,
    "photo": PHOTO_LM_NEIGHBOR,
}


#############################################################################
################### LLM Direct Inference (GNN Similar Nodes) ################
#############################################################################

CORA_GNN_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_GNN_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
INSTAGRAM_GNN_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
PUBMED_GNN_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
WIKICS_GNN_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
REDDIT_GNN_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """
PHOTO_GNN_NEIGHBOR = "Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance and structural information) of its 1st-order neighbors. The relation between the node and its neighbors is 'co-purchase'. Question: Which of the following type does this photo item belong to? Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography. Reply only one category that you think this item might belong to. Only reply the category name I give without any other words.\n\nAnswer: """

GNN_NEIGHBOR_PROMPTS = {
    "cora": CORA_GNN_NEIGHBOR,
    "citeseer": CITESEER_GNN_NEIGHBOR,
    "instagram": INSTAGRAM_GNN_NEIGHBOR,
    "pubmed": PUBMED_GNN_NEIGHBOR,
    "wikics": WIKICS_GNN_NEIGHBOR,
    "reddit": REDDIT_GNN_NEIGHBOR,
    "photo": PHOTO_GNN_NEIGHBOR,
}


#############################################################################
################### LLM Direct Inference (w. Summary) #######################
#############################################################################

CORA_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
INSTAGRAM_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
PUBMED_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
WIKICS_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
REDDIT_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this user might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """
PHOTO_LLM_NEIGHBOR = "Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'co-purchase'. Question: Which of the following type does this photo item belong to? Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography. Reply only one category that you think this item might belong to. Only reply the category name I give without any other words.\n\nAnswer: """

LLM_NEIGHBOR_PROMPTS = {
    "cora": CORA_LLM_NEIGHBOR,
    "citeseer": CITESEER_LLM_NEIGHBOR,
    "instagram": INSTAGRAM_LLM_NEIGHBOR,
    "pubmed": PUBMED_LLM_NEIGHBOR,
    "wikics": WIKICS_LLM_NEIGHBOR,
    "reddit": REDDIT_LLM_NEIGHBOR,
    "photo": PHOTO_LLM_NEIGHBOR,
}


#############################################################################
################### LLM Direct Inference (Chain of Thought) #################
#############################################################################

CORA_COT = """Here I give you the content of the node itself. 
Question: Based on this information, which of the following sub-categories of AI does this paper (this node) belong to? 
Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods.

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category.
Output format: <reason: >, <classification: >
"""

CITESEER_COT = """Here I give you the content of the node itself. 
Question: Which of the following themes does this paper belong to? 
Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence).

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category
Output format: <reason: >, <classification: >
"""

INSTAGRAM_COT = """Here I give you the content of the node itself. 
Question: Which of the following categories does this Instagram user belong to? 
Here are the 2 categories: Normal Users, Commercial Users.

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category.
Output format: <reason: >, <classification: >
"""

PUBMED_COT = """Here I give you the content of the node itself. 
Question: Which of the following topics does this scientific publication talk about? 
Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2.

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category.
Output format: <reason: >, <classification: >
"""

WIKICS_COT = """Here I give you the content of the node itself. 
Question: Which of the following branches of Computer Science does this Wikipedia-based dataset belong to? 
Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics.

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category.
Output format: <reason: >, <classification: >
"""

REDDIT_COT = """Here I give you the content of the node itself. 
Question: Which of the following categories does this Reddit user belong to? 
Here are the 2 categories: Normal Users, Popular Users.

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category.
Output format: <reason: >, <classification: >
"""

PHOTO_COT = """Here I give you the content of the node itself. 
Question: Which of the following types does this photo item belong to? 
Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography.

Answer: Let's think about it step by step. Analyze the content of the node and choose one appropriate category.
Output format: <reason: >, <classification: >
"""

COT_PROMPTS = {
    "cora": CORA_COT,
    "citeseer": CITESEER_COT,
    "instagram": INSTAGRAM_COT,
    "pubmed": PUBMED_COT,
    "wikics": WIKICS_COT,
    "reddit": REDDIT_COT,
    "photo": PHOTO_COT,
}


#############################################################################
################### LLM Direct Inference (Tree of Thought) ##################
#############################################################################

CORA_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following sub-categories of AI does this paper (this node) belong to? 
Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods.

Answer: Let's think through this using a tree of thought approach.
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

CITESEER_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following theme does this paper belong to? 
Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence).

Answer: Let's think through this using a tree of thought approach. 
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

INSTAGRAM_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following categories does this Instagram user belong to? 
Here are the 2 categories: Normal Users, Commercial Users.

Answer: Let's think through this using a tree of thought approach. 
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

PUBMED_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following topic does this scientific publication talk about? 
Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2.

Answer: Let's think through this using a tree of thought approach. 
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

WIKICS_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following branch of Computer Science does this Wikipedia-based dataset belong to? 
Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics.

Answer: Let's think through this using a tree of thought approach. 
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

REDDIT_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following categories does this Reddit user belong to? 
Here are the 2 categories: Normal Users, Popular Users.

Answer: Let's think through this using a tree of thought approach. 
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

PHOTO_TOT = """Here I give you the content of the node itself. 
Imagine three different experts are answering this question. 
All experts will write down 1 step of their thinking, then share it with the group. 
Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
Question: Based on this information, which of the following type does this photo item belong to? 
Here are the 12 categories: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography.

Answer: Let's think through this using a tree of thought approach. 
Output format: <discussion: >, <classification: >. The classification should only consist one of the category name I listed.
"""

TOT_PROMPTS = {
    "cora": CORA_TOT,
    "citeseer": CITESEER_TOT,
    "instagram": INSTAGRAM_TOT,
    "pubmed": PUBMED_TOT,
    "wikics": WIKICS_TOT,
    "reddit": REDDIT_TOT,
    "photo": PHOTO_TOT,
}


#############################################################################
####################### LLM Direct Inference (ReACT) ########################
#############################################################################

CORA_REACT = """Here I give you the content of the node itself. Your task is to determine which of the following sub-categories of AI this paper belongs to: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. 
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

CITESEER_REACT = """Here I give you the content of the node itself. Your task is to determine which of the following themes this paper belongs to: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). 
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

INSTAGRAM_REACT = """Here I give you the content of the node itself. Your task is to determine whether this Instagram user belongs to Normal Users or Commercial Users.
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

PUBMED_REACT = """Here I give you the content of the node itself. Your task is to determine which of the following topics this scientific publication talks about: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2.
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

WIKICS_REACT = """Here I give you the content of the node itself. Your task is to determine which branch of Computer Science this Wikipedia-based dataset belongs to: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics.
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

REDDIT_REACT = """Here I give you the content of the node itself. Your task is to determine whether this Reddit user belongs to Normal Users or Popular Users. Popular Users' posted content is often more attractive.
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

PHOTO_REACT = """Here I give you the content of the node itself. Your task is to determine which of the following types this photo item belongs to: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography.
Solve this question by interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be one of the following:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
The output format must be <process: >, <classification: >. The classification should only consist one of the category name I listed."""

REACT_PROMPTS = {
    "cora": CORA_REACT,
    "citeseer": CITESEER_REACT,
    "instagram": INSTAGRAM_REACT,
    "pubmed": PUBMED_REACT,
    "wikics": WIKICS_REACT,
    "reddit": REDDIT_REACT,
    "photo": PHOTO_REACT,
}
