from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Store the cookie consent status in memory for simplicity (not persistent)
consent_status = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/set_cookie_consent', methods=['POST'])
def set_cookie_consent():
    global consent_status
    data = request.get_json()
    consent = data.get('consent')
    consent_status['user_consent'] = consent
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True, port=1234)
