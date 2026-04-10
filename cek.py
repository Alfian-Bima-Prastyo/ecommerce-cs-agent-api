import google.generativeai as genai

genai.configure(api_key="AIzaSyALFGNKbseJUIGQlZENTPbaRv2gZefaON8")

for model in genai.list_models():
    print(model.name)