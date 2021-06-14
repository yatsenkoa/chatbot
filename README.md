# Chatbot



### TODO
- make model actually work
- move model to python script, add separate utils file for readability
- make chatbot demo
- make closed domain responses work
- make responses from word bank word with decoder

A chatbot written in pytorch. This bot is meant to be something in between a closed and open-domain chatbot - it can have predefined responses, or use an RNN to generate responses 

### Why use closed-domain instead of open-domain?
- open-domain bots are good at inference and human-like conversation, but take a lot of data to train. Closed-domain chatbots require only the pre-recorded responses.

# Notes

### Architecture
- GRU encoder and decoder, for now

