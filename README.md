# Healthcare Chatbot

## About the Project
## AIM:
The prime Objective of the project is to develop a chatbot which interacts with the user in question-answer format to provide the required personalized and reliable healthcare information and support.
## DESCRIPTION:
The HealthBuddy Chatbot aims to create a versatile chatbot that can offer assistance in various aspects of healthcare, including symptom diagnosis, mental health consultation, nutrition guidance, and more. The inspiration behind this project is to empower users to make informed healthcare decisions and promote overall well-being.
This objective is satisfied by fine-tuning an already existing LLM on a medical-specific dataset.

## TECH STACK
#### We have used the following technologies for this project:
- [Python](https://www.python.org/)
- [Numpy](https://numpy.org/doc/#)
- [Pytorch](https://pytorch.org/)
- [Google Colab](https://colab.research.google.com/)

## File Structure


## MODELS USED
- [Falcon-7b](https://huggingface.co/vilsonrodrigues/falcon-7b-instruct-sharded) for fine tuning 
- [Llama-2-chat](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) for fine tuning 
- [dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b) for Document based LLM

## THEORY AND APPROACH
### Terms/prerequsites required 
### NLP:
Natural language processing (NLP) refers to the branch of computer science—and more specifically,
the branch of artificial intelligence or AI—concerned with giving computers the ability to understand
text and spoken words in much the same way human beings can.This technologies enable computers
to process human language in the form of text or voice data and to ‘understand’ its full meaning,
complete with the speaker or writer’s intent and sentiment.

### Finetuning LLMs:
- Pre-trained models like GPT-3 have been trained on massive datasets to learn general linguistic skills and knowledge. This gives them strong capabilities out of the box.
- However, their knowledge is still general. To adapt them to specialized domains and tasks, we can fine-tune the models on smaller datasets specific to our needs.

[Fine-tuning](https://www.labellerr.com/blog/content/images/size/w2000/2023/08/6488323befb01b8fac0fe171_VmVuJPKbeUxwrJcqoQ5EYZOSWGiW2rE-C_Yj563jJAQrE2V8PP1ibzWUrXrDLXzJIl7i205vzAfQKRL53whzjrBJKXtP8J9j4J_Pn9vtAh-o9sxEUAIPxHYZgNuJyvOXleZZDzTxr8sIh371Xznqwn8.png)

- One of the major disadvantages of finetuning is catastrophic forgetting - Catastrophic forgetting occurs due to the nature of the optimization process during training. When a model trains to minimize the current task’s loss, it adjusts its parameters to better fit the new data. However, this adjustment often results in the model deviating from its original state, leading to a loss of knowledge encoded in its weights.

**Parameter-Efficient-Finetuning**
- Parameter Efficient Fine-Tuning (PEFT) methods specifically attempt to address some of the challenges of performing full fine-training. 
- PEFT updates only a small subset of parameters which helps prevent catastrophic forgetting.The new parameters are combined with the original LLM weights for inference.

![Fine=tuning](Healthcare-Chatbot\assets\Flowchart1.jpeg)

### Doc-based LLMs
![Doc-based](Healthcare-Chatbot\assets\Flowchart2.jpeg)

## FUTURE WORKS
1. Training the model on a larger dataset provided we have access to local GPU for more accurate results
2. Creating a proper user-friendly interface.
3. Providing contact information for appropriate specialists for consultancy.


## Contributors
* [Anushka Yadav](https://github.com/2412anushka)
* [Raya Chakravarty](https://github.com/Raya679)
* [Tvisha Vedant](https://github.com/tvilight4)

## AKNOWLEGEMENT AND RESOURCES
*  Special thanks to [COC VJTI](https://github.com/CommunityOfCoders) for [ProjectX](https://github.com/CommunityOfCoders/Project-X-2023) 2023 
* Referred [this](https://www.youtube.com/watch?v=EMXfZB8FVUA) for understanding the use of pytorch
* Completed [these](https://www.coursera.org/specializations/deep-learning) 3 courses for understanding Deep Learning concepts like RNNs, LSTMs and Attention Mechanism and learnt to make a DL model.
* Referred [this](https://arxiv.org/pdf/1706.03762.pdf) for understanding of Transformer Architecture
* Completed [this](https://www.coursera.org/learn/generative-ai-with-llms) course for learning the concept of Finetuning.
* Special Thanks to our awesome mentors [Dishie Vinchhi](https://github.com/Dishie2498) and [Om Doiphode](https://github.com/Om-Doiphode) who always helped us during our project journey. 
