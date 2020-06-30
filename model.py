# Load a spacy model and chekc if it has ner
import spacy
nlp=spacy.load('en_core_web_sm')
# en_core_web_sm
print(nlp.pipe_names)
# nlp.add_pipe()
#print(nlp)

# Performing NER on E-commerce article

article_text="""India that previously comprised only a handful of players in the e-commerce space, is now home to many biggies and giants battling out with each other to reach the top. This is thanks to the overwhelming internet and smartphone penetration coupled with the ever-increasing digital adoption across the country. These new-age innovations not only gave emerging startups a unique platform to deliver seamless shopping experiences but also provided brick and mortar stores with a level-playing field to begin their online journeys without leaving their offline legacies.
In the wake of so many players coming together on one platform, the Indian e-commerce market is envisioned to reach USD 84 billion in 2021 from USD 24 billion in 2017. Further, with the rate at which internet penetration is increasing, we can expect more and more international retailers coming to India in addition to a large pool of new startups. This, in turn, will provide a major Philip to the organized retail market and boost its share from 12% in 2017 to 22-25% by 2021. 
Here’s a view to the e-commerce giants that are dominating India’s online shopping space:
Amazon – One of the uncontested global leaders, Amazon started its journey as a simple online bookstore that gradually expanded its reach to provide a large suite of diversified products including media, furniture, food, and electronics, among others. And now with the launch of Amazon Prime and Amazon Music Limited, it has taken customer experience to a godly level, which will remain undefeatable for a very long time. 
Flipkart – Founded in 2007, Flipkart is recognized as the national leader in the Indian e-commerce market. Just like Amazon, it started operating by selling books and then entered other categories such as electronics, fashion, and lifestyle, mobile phones, etc. And now that it has been acquired by Walmart, one of the largest leading platforms of e-commerce in the US, it has also raised its bar of customer offerings in all aspects and giving huge competition to Amazon. 
Snapdeal – Started as a daily deals platform in 2010, Snapdeal became a full-fledged online marketplace in 2011 comprising more than 3 lac sellers across India. The platform offers over 30 million products across 800+ diverse categories from over 125,000 regional, national, and international brands and retailers. The Indian e-commerce firm follows a robust strategy to stay at the forefront of innovation and deliver seamless customer offerings to its wide customer base. It has shown great potential for recovery in recent years despite losing Freecharge and Unicommerce. 
ShopClues – Another renowned name in the Indian e-commerce industry, ShopClues was founded in July 2011. It’s a Gurugram based company having a current valuation of INR 1.1 billion and is backed by prominent names including Nexus Venture Partners, Tiger Global, and Helion Ventures as its major investors. Presently, the platform comprises more than 5 lac sellers selling products in nine different categories such as computers, cameras, mobiles, etc. 
Paytm Mall – To compete with the existing e-commerce giants, Paytm, an online payment system has also launched its online marketplace – Paytm Mall, which offers a wide array of products ranging from men and women fashion to groceries and cosmetics, electronics and home products, and many more. The unique thing about this platform is that it serves as a medium for third parties to sell their products directly through the widely-known app – Paytm. 
Reliance Retail – Given Reliance Jio’s disruptive venture in the Indian telecom space along with a solid market presence of Reliance, it is no wonder that Reliance will soon be foraying into retail space. As of now, it has plans to build an e-commerce space that will be established on online-to-offline market program and aim to bring local merchants on board to help them boost their sales and compete with the existing industry leaders. 
Big Basket – India’s biggest online supermarket, Big Basket provides a wide variety of imported and gourmet products through two types of delivery services – express delivery and slotted delivery. It also offers pre-cut fruits along with a long list of beverages including fresh juices, cold drinks, hot teas, etc. Moreover, it not only provides farm-fresh products but also ensures that the farmer gets better prices. 
Grofers – One of the leading e-commerce players in the grocery segment, Grofers started its operations in 2013 and has reached overwhelming heights in the last 5 years. Its wide range of products includes atta, milk, oil, daily need products, vegetables, dairy products, juices, beverages, among others. With its growing reach across India, it has become one of the favorite supermarkets for Indian consumers who want to shop grocery items from the comforts of their homes. 
Digital Mall of Asia – Going live in 2020, Digital Mall of Asia is a very unique concept coined by the founders of Yokeasia Malls. It is designed to provide an immersive digital space equipped with multiple visual and sensory elements to sellers and shoppers. It will also give retailers exclusive rights to sell a particular product category or brand in their respective cities. What makes it unique is its zero-commission model enabling retailers to pay only a fixed amount of monthly rental instead of paying commissions. With its one-of-a-kind features, DMA is expected to bring
never-seen transformation to the current e-commerce ecosystem while addressing all the existing e-commerce worries such as counterfeiting. """

doc=nlp(article_text)
for ent in doc.ents:
  print(ent.text,ent.label_)


# Load pre-existing spacy model
import spacy
nlp=spacy.load('en_core_web_sm')

# Getting the pipeline component
ner=nlp.get_pipe("ner")


# training data
TRAIN_DATA = [
              ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
              ("I reached Chennai yesterday.", {"entities": [(19, 28, "GPE")]}),
              ("I recently ordered a book from Amazon", {"entities": [(24,32, "ORG")]}),
              ("I was driving a BMW", {"entities": [(16,19, "PRODUCT")]}),
              ("I ordered this from ShopClues", {"entities": [(20,29, "ORG")]}),
              ("Fridge can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]}),
              ("I bought a new Washer", {"entities": [(16,22, "PRODUCT")]}),
              ("I bought a old table", {"entities": [(16,21, "PRODUCT")]}),
              ("I bought a fancy dress", {"entities": [(18,23, "PRODUCT")]}),
              ("I rented a camera", {"entities": [(12,18, "PRODUCT")]}),
              ("I rented a tent for our trip", {"entities": [(12,16, "PRODUCT")]}),
              ("I rented a screwdriver from our neighbour", {"entities": [(12,22, "PRODUCT")]}),
              ("I repaired my computer", {"entities": [(15,23, "PRODUCT")]}),
              ("I got my clock fixed", {"entities": [(16,21, "PRODUCT")]}),
              ("I got my truck fixed", {"entities": [(16,21, "PRODUCT")]}),
              ("Flipkart started it's journey from zero", {"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Max", {"entities": [(24,27, "ORG")]}),
              ("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Swiggy", {"entities": [(24,29, "ORG")]})
              ]

# Adding labels to the `ner`

for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(30):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
        print("Losses", losses)

# # Testing the model
# doc = nlp("I was driving a Suzuki")
# print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# doc1 = nlp("I was driving a Maggi")
# print("Entities", [(ent.text, ent.label_) for ent in doc1.ents])

# Save the  model to directory
output_dir = Path('../Product-Keyword-Extraction-with-NLP/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)