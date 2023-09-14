
graphing_bot_colors = {
    'Blender2': 'purple',
    'BART-FiD-RAG': 'royalblue',
    'Emora': 'turquoise',
    'Blender-Decode': 'green',

    'first': 'lightgray',
    'second': 'dimgray'
}

graphing_botpair_colors = {
    ('Blender2', 'Emora'): 'red',
    ('BART-FiD-RAG', 'Blender2'): 'orange',
    ('BART-FiD-RAG', 'Blender-Decode'): 'blue',
    ('BART-FiD-RAG', 'Emora'): 'black',
    ('Blender-Decode', 'Blender2'): 'green',
    ('Blender-Decode', 'Emora'): 'brown'
}

graphing_dim_colors = {
    'consistent': 'red',
    'engaging': 'green',
    'emotional': 'blue',
    'grammatical': 'purple',
    'proactive': 'brown',
    'quality': 'black',
    'informative': 'orange',
    'relevant': 'turquoise'
}

bot_transformer = {
    'blender2_3B': 'Blender2',
    'emora': 'Emora',
    'rerank_blender': 'Blender-Decode',
    'bart_fid_rag_bcb': 'BART-FiD-RAG'
}

bot_tags = {
    'Blender2': 'B2',
    'BART-FiD-RAG': 'BFR',
    'Emora': 'EM',
    'Blender-Decode': 'BD'
}

dimensions_identity = {
    'consistent': 'consistent',
    'emotional': 'emotional',
    'engaging': 'engaging',
    'grammatical': 'grammatical',
    'informative': 'informative',
    'proactive': 'proactive',
    'quality': 'quality',
    'relevant': 'relevant'
}

dimensions_transformer = {
    'grammatical': 'Gra',
    'quality': 'Qua',
    'engaging': 'Eng',
    'proactive': 'Pro',
    'informative': 'Inf',
    'relevant': 'Rel',
    'consistent': 'Con',
    'emotional': 'Emo',
    'topic switch': 'Top',
    'incorrect fact': '!Fac',
    'life info': 'Lif',
    'correct fact': 'Fac',
    'ignore': 'Ign',
    'antisocial': '!Soc',
    'redundant': 'Red',
    'empathetic': 'Emp',
    'follow up': 'Fol',
    'preference info': 'Pre',
    'irrelevant': '!Rel',
    'commonsense contradiction': '!Com',
    'partner contradiction': '!Par',
    'lack of empathy': '!Emp',
    'self contradiction': '!Sel',
    'uninterpretable': '!Int'
}