from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form.get("topic")
        temperature = float(request.form.get("temperature"))

        response = requests.post("http://localhost:8000/generate_report/",
                                 json={"topic": topic, "temperature": temperature})

        if response.status_code == 200:
            report = response.json().get("report")
            return render_template("index.html", report=report, topic=topic)
        else:
            error = response.json().get("detail")
            return render_template("index.html", error=error)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
