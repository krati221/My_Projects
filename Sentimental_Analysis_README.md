🚀 **Sentiment Analysis Project: From Classical NLP to Transformers**

I recently built a complete **Sentiment Analysis pipeline** using both traditional NLP techniques and modern deep learning models to understand customer opinions from real-world data.

🔍 **Dataset Used:**
Amazon Fine Food Reviews (real customer feedback)

---

💡 **What I Did in This Project**

📊 **1. Data Analysis & Visualization (EDA)**

* Explored review distribution across ratings
* Visualized how customer sentiment varies with star ratings
* Used Seaborn & Matplotlib for insights

---

🧠 **2. NLP with NLTK (VADER Model)**

* Applied **VADER Sentiment Analyzer** (Bag-of-Words approach)
* Extracted:

  * Positive, Neutral, Negative scores
  * Compound sentiment score
* Compared sentiment vs actual review ratings

---

🤖 **3. Advanced NLP with Transformers (RoBERTa)**

* Used pretrained model: `cardiffnlp/twitter-roberta-base-sentiment`
* Implemented tokenizer + model inference
* Applied **softmax** to get probability scores
* Captured context-aware sentiment (better than VADER)

---

⚖️ **4. Model Comparison (VADER vs RoBERTa)**

* Combined both results into a single dataset
* Visualized relationships using **pairplots**
* Observed how:

  * VADER works well for simple text
  * RoBERTa performs better on complex/contextual sentences

---

📈 **Key Insights**

* Traditional NLP (VADER) is fast and interpretable
* Transformer models (RoBERTa) provide deeper contextual understanding
* Combining both gives better analytical power

---

🛠️ **Tools & Technologies**

* Python, Pandas, NumPy
* NLTK (VADER)
* HuggingFace Transformers
* Seaborn, Matplotlib

---

✨ **What I Learned**

* Difference between rule-based vs deep learning NLP
* Importance of context in sentiment analysis
* How to build a complete NLP pipeline from scratch

---
We will be doing some sentiment analysis in python using two differ techniques:
 1) VADER - Valence aware dictionary and sentiment reasoner - bag of words approach 
 2) Roberta Pretrained model from hugging face 
 3) Hugging face pipeline

Errors I got during projects -
1)set.xlables - set.xlabel
2)nltk - ntlk  
3)Code-  nltk.pos_tag(tokens)
ERROR - LookupError                               Traceback (most recent call last)
/tmp/ipykernel_55/2138528135.py in <cell line: 0>()
----> 1 nltk.pos_tag(tokens) #Error

/usr/local/lib/python3.12/dist-packages/nltk/tag/__init__.py in pos_tag(tokens, tagset, lang)
    166     :rtype: list(tuple(str, str))
    167     """
--> 168     tagger = _get_tagger(lang)
    169     return _pos_tag(tokens, tagset, tagger, lang)
    170 

/usr/local/lib/python3.12/dist-packages/nltk/tag/__init__.py in _get_tagger(lang)
    108         tagger = PerceptronTagger(lang=lang)
    109     else:
--> 110         tagger = PerceptronTagger()
    111     return tagger
    112 

/usr/local/lib/python3.12/dist-packages/nltk/tag/perceptron.py in __init__(self, load, lang)
    181         self.classes = set()
    182         if load:
--> 183             self.load_from_json(lang)
    184 
    185     def tag(self, tokens, return_conf=False, use_tagdict=True):

/usr/local/lib/python3.12/dist-packages/nltk/tag/perceptron.py in load_from_json(self, lang)
    271     def load_from_json(self, lang="eng"):
    272         # Automatically find path to the tagger if location is not specified.
--> 273         loc = find(f"taggers/averaged_perceptron_tagger_{lang}/")
    274         with open(loc + TAGGER_JSONS[lang]["weights"]) as fin:
    275             self.model.weights = json.load(fin)

/usr/local/lib/python3.12/dist-packages/nltk/data.py in find(resource_name, paths)
    577     sep = "*" * 70
    578     resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
--> 579     raise LookupError(resource_not_found)
    580 
    581 

LookupError: 
**********************************************************************
  Resource averaged_perceptron_tagger_eng not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('averaged_perceptron_tagger_eng')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load taggers/averaged_perceptron_tagger_eng/

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************
4)CODE-entities = nltk.chunk.ne_chunk(tagged)
       entities.pprint()
  Error-LookupError                               Traceback (most recent call last)
/tmp/ipykernel_55/3092750047.py in <cell line: 0>()
----> 1 entities = nltk.chunk.ne_chunk(tagged)
      2 entities.pprint()

/usr/local/lib/python3.12/dist-packages/nltk/chunk/__init__.py in ne_chunk(tagged_tokens, binary)
    190         chunker = ne_chunker(fmt="binary")
    191     else:
--> 192         chunker = ne_chunker()
    193     return chunker.parse(tagged_tokens)
    194 

/usr/local/lib/python3.12/dist-packages/nltk/chunk/__init__.py in ne_chunker(fmt)
    172     Load NLTK's currently recommended named entity chunker.
    173     """
--> 174     return Maxent_NE_Chunker(fmt)
    175 
    176 

/usr/local/lib/python3.12/dist-packages/nltk/chunk/named_entity.py in __init__(self, fmt)
    327 
    328         self._fmt = fmt
--> 329         self._tab_dir = find(f"chunkers/maxent_ne_chunker_tab/english_ace_{fmt}/")
    330         self.load_params()
    331 

/usr/local/lib/python3.12/dist-packages/nltk/data.py in find(resource_name, paths)
    577     sep = "*" * 70
    578     resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
--> 579     raise LookupError(resource_not_found)
    580 
    581 

LookupError: 
**********************************************************************
  Resource maxent_ne_chunker_tab not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('maxent_ne_chunker_tab')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load chunkers/maxent_ne_chunker_tab/english_ace_multiclass/

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************     
FIX--nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

CODE - Model = f"FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)
ERROR--Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
RobertaForSequenceClassification LOAD REPORT from: FacebookAI/roberta-base
Key                             | Status     | 
--------------------------------+------------+-
lm_head.bias                    | UNEXPECTED | 
lm_head.dense.weight            | UNEXPECTED | 
roberta.embeddings.position_ids | UNEXPECTED | 
lm_head.layer_norm.weight       | UNEXPECTED | 
lm_head.dense.bias              | UNEXPECTED | 
lm_head.layer_norm.bias         | UNEXPECTED | 
classifier.out_proj.bias        | MISSING    | 
classifier.dense.weight         | MISSING    | 
classifier.out_proj.weight      | MISSING    | 
classifier.dense.bias           | MISSING    | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
After changing code - Model = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
Error-RobertaForSequenceClassification LOAD REPORT from: cardiffnlp/twitter-roberta-base-sentiment
Key                             | Status     |  | 
--------------------------------+------------+--+-
roberta.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch. (we can ignore it )
