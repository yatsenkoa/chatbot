# Chatbot

A chatbot written in pytorch. This bot is meant to be something in between a closed and open-domain chatbot - it can have predefined responses for some area in the latent space


### Why use closed-domain instead of open-domain?
- open-domain bots are good at inference and human-like conversation, but take a lot of data to train. Closed-domain chatbots require only the pre-recorded responses, but are less 'smart'



### How do the predefined responses work?
- Predefined responses are kept in a 


### Technical
- When selecting a response, we can scan the question bank and compare the similarity of the latent space of the question with both the generated response and the prerecorded questions, picking the better match



### Basic seq2seq bot in pytorch
- pretrained word embeddings
- LSTM


### Roadmap
- [ ] Figure out how to make a seq2seq open domain chatbot
- [ ] Figure out how to make a closed-domain chatbot with good storage
- [ ] Figure out how to combine the two


### Todo seq2seq


