from app import train_classifier, classify_text

def test_query():
    text = input("Hi I'm a chatbot. Lets talk: ")
    result = classify_text(text, 'pro')
    print(result)

def run_app():
    option = input("1 - Interact | 2 - Train | 3 - Exit: ")

    if option == "1":
        test_query()
    elif option == "2":
        model = input("Model to train: 1 - Base | 2 - Pro: ")
        selected_model = 'base' if model == '1' else 'pro'
        train_classifier(selected_model)
    elif option == "3":
        return False

    return True

while True:
    if not run_app():
        break