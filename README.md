# HealthBuddy - Your AI Health Companion
## About the Project
## AIM:
#### The prime Objective of the project is to develop a chatbot which interacts with the user in questionanswer format to provide the required personalized and reliable healthcare information and support.This objective is satisfied by the use of transformer architecture which is a Neural Network model(ML model) for performing NLP related tasks.The use of transformer architecture enhances thefunctionality of the chatbot model.
## DESCRIPTION:
#### The HealthBuddy Chatbot is an innovative and user-friendly healthcare solution designed to provide individuals with personalized and reliable healthcare information and support. This project aims to create a versatile chatbot that can offer assistance in various aspects of healthcare, including symptom diagnosis, mental health consultation, nutrition guidance, and more. The inspiration behind this project is to empower users to make informed healthcare decisions and promote overall well-being.
### Monitoring: 
Awareness and tracking of user’s behavior, anxiety, and weight changes to encourage developing better habits.
### Anonymity: 
Especially in sensitive and mental health issues.
### Personalization: 
Level of personalization depends on the specific application. Some applications make use of measurements of:
- Physical vitals (oxygenation, heart rhythm, body temperature) via mobile sensors.
- Patient behavior via facial recognition.
### Real time interaction: 
Immediate response, notifications, and reminders.
### Scalability: 
Ability to react with numerous users at the same time
## TECH STACK
#### We have used the following technologies for this project:
- Python
- Pytorch framework
- Google Colab

## MODELS USED
- Falcon-7b for fine tuning of pretrained model
- dolly-v2-3b for Document based LLM

## THEORY AND APPROACH
### Terms/prerequsites required 
### NLP:
Natural language processing (NLP) refers to the branch of computer science—and more specifically,
the branch of artificial intelligence or AI—concerned with giving computers the ability to understand
text and spoken words in much the same way human beings can.This technologies enable computers
to process human language in the form of text or voice data and to ‘understand’ its full meaning,
complete with the speaker or writer’s intent and sentiment.
### Attention mechanism:
The memory of RNNs is low, and the size of the reference window (the number of words before the
current word, from which RNNs can draw contextual information ) is small. This gave rise to the use of
LSTMs but still they had a limited reference window, even though larger than RNNs. the power of the
attention mechanism is that it doesn’t suffer from short term memory. The attention mechanism, in
theory, and given enough compute resources, have an infinite window to reference from.The Attention
mechanism enables the transformers to have extremely long term memory.
The attention mechanism’s power was demonstrated in the paper “Attention Is All You Need”, where
the authors introduced a new novel neural network called the Transformers which is an attention-based
encoder-decoder type architecture.
### Transformer architecture:
The gist about tranformers is given above.
Components:
Input Embedding:
When each word is fed into the network, this code will perform a look-up and retrieve its embedding
vector. These vectors will then be learnt as a parameters by the model,

Positional encoding:
The inputed word gets a context(meaning of word in inputed sentence and position in the sentence)
because of this process
For every odd index on the input vector, create a vector using the cos function. For every even index,
create a vector using the sin function. Then add those vectors to their corresponding input embeddings.
This successfully gives the network information on the position of each vector. The sin and cosine
functions were chosen in tandem because they have linear properties the model can easily learn to
attend to.
Encoder Layer
It maps all the input sequences into a continuous representation that holds the learned information for
that entire sequence.
Multi-Headed Attention:

Multi-headed attention in the encoder applies a specific attention mechanism called self-attention. Self-
attention allows the models to associate each word in the input, to other words enables the model to

respond appropriately.

Diagram of transformer architecture:

![LOC](https://madewithml.com/static/images/foundations/transformers/architecture.png)

Query, Key, and Value Vectors
To achieve self-attention, we feed the input into 3 distinct fully connected layers to
create the query, key, and value vectors. In the case of the Encoder, V, K and Q vectors will simply be
identical copies of the embedding vector (plus positional encoding). queries and keys undergo a dot
product matrix multiplication to produce a score matrix, which has each word related to the other word.
This is followed by scalling down and softmax which makes the model more confident.In a nutshell,
the multiheaded attention helps to give relevant outputs by mapping related words.
The Residual Connections, Layer Normalization, and Feed Forward Network are the further steps
The decoder’s job is to generate text sequences. The decoder has a similar sub-layer as the encoder.

The multiheaded attention feature of the transformer is one of the main features why they are used for
developing tools like chatbots.
### Steps to implement chatbot:
#### The following tasks are brought to reality in code by using the pytorch framework which is an opensource deep learning framework created by meta(based on LUA based torch library).In PyTorch, thedata that has to be processed is input in the form of a tensor.
- insatll pytorch and other required libraries
- consider various use cases for data accumulation
- create/use already available training data in a json file
- train the transformer model
- implement the transformer model
- the model will be processed and trained using available data
- implement the chat : load the trained model and make predictions
## FUTURE PROSPECTS
The chatbot can help a general physician read blood reports and identify disease or based on
symptoms
It can recommend medications based on disease and symptoms to help the operation of hospitals and
medical clinics
A standard operating procedure for hospitals and therapy centers can be developed by deploying the
chatbot model on cloud.

## RESOURCES
- [Linear Algebra playlist](https://www.youtube.com/watch?v=fNk_zzaMoSs)
- [Courses 1,2 and 5 from Coursera](https://www.coursera.org/specializations/deep-learning?utm_medium=sem&utm_source=gg&utm_campaign=B2C_INDIA_deep-learning_deeplearning-ai_FTCOF_specializations_arte-agency&campaignid=20400873830&adgroupid=150410452254&device=c&keyword=coursera%20deep%20learning&matchtype=b&network=g&devicemodel=&adposition=&creativeid=666993398104&hide_mobile_promo&gclid=CjwKCAjw15eqBhBZEiwAbDomEukwcMqt8-EzcLZj3XT4hjju_dcT32n1RwBJVKQMYGit3mp4G5HKLRoCK3gQAvD_BwE)
- [Pytorch framework by Patrick Loeber (playlist)](https://www.youtube.com/watch?v=EMXfZB8FVUA)
- [Document based LLM](https://medium.com/@Siddharth.jh/conversational-chat-bot-using-open-source-llm-model-dolly-2-0-with-added-memory-acfacc13a69e)
- [Attention Is All You Need (original transformer reasearch paper by Vaswani)](https://arxiv.org/pdf/1706.03762.pdf)
- [ChatDoctor reasearch paper](https://arxiv.org/ftp/arxiv/papers/2303/2303.14070.pdf)
- [ChatDoctor GitHub repo](https://github.com/Kent0n-Li/ChatDoctor)
- [Coursera Geneartive AI course](https://www.coursera.org/learn/generative-ai-with-llms)
