def test_query():
    text = input("Hi I'm a chatbot. Lets talk: ")
    print("You said: " + text)

def run_app():
    option = input("Enter 1 to interact or 2 to exit: ")

    if option == "1":
        test_query()
    elif option == "2":
        return False

    return True

while True:
    if not run_app():
        break