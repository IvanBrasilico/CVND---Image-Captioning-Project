import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = None
    
    
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        output, _ = self.lstm(embeddings)
        outputs = self.linear(output)
        return outputs
    
    
    def sample(self, inputs, states=None, max_len=20):
        """ Accepts pre-processed image tensor (inputs) and
        returns predicted sentence (list of tensor ids of length max_len)"""
        sampled_ids = []
        batch_size = inputs.shape[0]
        print("Batch_size  " , batch_size)
        for i in range(max_len):
            lstm_states, _ = self.lstm(inputs) 
            outputs = self.linear(lstm_states.squeeze(1)) 
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.cpu().numpy()[0].item())
            if predicted == 1:
                break
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids
    
