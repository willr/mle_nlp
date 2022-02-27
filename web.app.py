from webapp.textsimilar import create_app, Environment

app = create_app(Environment.DEVELOPMENT)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)