# cell 1
model = GPTModel(GPT_CONFIG_124M)
torch.save(model.state_dict(), "model.pth")

# cell 2
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# cell 3
model1 = GPTModel(GPT_CONFIG_124M)
model1.load_state_dict(torch.load("model.pth"))
model1.eval()
