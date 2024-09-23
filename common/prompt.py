CORA_Q = """Question: Which of the following sub-categories of AI does this paper belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
PUBMED_Q = """Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_Q = """Question: Which of the following theme does this paper belong to? Here are the 6 categories: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence). Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: """
WIKICS_Q = """Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: """
ARXIV_Q = """Question: Which of the following 
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
        Only reply the category name (not the explanation) I given without any other words, please don't use your own words.
        Be careful, only use the name of the category I give you, not the explanation part or any other words.
        Answer:
    """
INSTAGRAM_Q = """Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this paper might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """
REDDIT_Q = """Question: Which of the following categories does this reddit user belong to? Here are the 2 categories: Normal Users, Popular Users. Popular Users' posted content are often more attractive. Reply only one category that you think this paper might belong to. Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.\n\nAnswer: """

ZEROSHOT_PROMPTS = {
    "cora": CORA_Q,
    "pubmed": PUBMED_Q,
    "citeseer": CITESEER_Q,
    "wikics": WIKICS_Q,
    "arxiv": ARXIV_Q,
    "instagram": INSTAGRAM_Q,
    "reddit": REDDIT_Q,
}


CORA_EXP = "Question: Which of the following sub-categories of AI does this paper belong to: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
PUBMED_EXP = "Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, give a detailed explanation with quotes from the text explaining why it is related to the chosen option.\n\nAnswer: "
ARXIV_EXP = "Question: Which arXiv CS subcategory does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form 'arxiv cs xx', and provide your reasoning.\n\nAnswer: "
CITESEER_EXP = "Question: Which of the following sub-categories of CS does this paper belong to: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence)? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
WIKICS_EXP = "Question: Which of the following sub-categories of CS does this Wikipedia page belong to: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
INSTAGRAM_EXP = "Question: Which of the following categories does this user on Instagram belong to:  Normal Users, Commercial Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
REDDIT_EXP = "Question: Which of the following categories does this user on Reddit belong to:  Normal Users, Popular Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "

EXPLANATION_PROMPTS = {
    "cora": CORA_EXP, 
    "pubmed": PUBMED_EXP, 
    "citeseer": CITESEER_EXP,
    "wikics": WIKICS_EXP,
    "arxiv": ARXIV_EXP,
    "instagram": INSTAGRAM_EXP,
    "reddit": REDDIT_EXP
}


CORA_RAW_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_RAW_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
INSTAGRAM_RAW_NEIGHBOR = """Here I give you the content of the node itself and the information of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this paper might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """

RAW_NEIGHBOR_PROMPTS = {
    "cora": CORA_RAW_NEIGHBOR,
    "citeseer": CITESEER_RAW_NEIGHBOR,
    "instagram": INSTAGRAM_RAW_NEIGHBOR,
}

CORA_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
INSTAGRAM_LM_NEIGHBOR = """Here I give you the content of the node itself and the information of its k-nearest neighbors (based on Euclidean Distance) of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this paper might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """

LM_NEIGHBOR_PROMPTS = {
    "cora": CORA_LM_NEIGHBOR,
    "citeseer": CITESEER_LM_NEIGHBOR,
    "instagram": INSTAGRAM_LM_NEIGHBOR,
}


CORA_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Based on these inforamtion, Which of the following sub-categories of AI does this paper(this node) belong to? Here are the 7 categories: Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
CITESEER_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'citation'. Question: Which of the following topic does this scientific publication talk about? Here are the 3 categories: Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2. Experimental category usually refers to Experimentally induced diabetes, Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1, Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2. Reply only one category that you think this paper might belong to. Only reply the category name without any other words.\n\nAnswer: """
INSTAGRAM_LLM_NEIGHBOR = """Here I give you the content of the node itself and the summary information of its 1st-order neighbors. The relation between the node and its neighbors is 'following'. Question: Which of the following categories does this instagram user belong to? Here are the 2 categories: Normal Users, Commercial Users. Reply only one category that you think this paper might belong to. Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.\n\nAnswer: """

LLM_NEIGHBOR_PROMPTS = {
    "cora": CORA_LLM_NEIGHBOR,
    "citeseer": CITESEER_LLM_NEIGHBOR,
    "instagram": INSTAGRAM_LLM_NEIGHBOR,
}