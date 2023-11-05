from app import train_classifier, classify_text

def test_query():
    text = input("Hi I'm a chatbot. Lets talk: ")
    result = classify_text(text)
    print(result)

def run_app():
    option = input("1 - Interact | 2 - Train | 3 - Exit: ")

    if option == "1":
        test_query()
    elif option == "2":
        train_classifier()
    elif option == "3":
        return False

    return True

while True:
    if not run_app():
        break