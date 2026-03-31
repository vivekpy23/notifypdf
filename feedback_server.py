from flask import Flask, request
app = Flask(__name__)

from db_manager import update_feedback

@app.route('/feedback', methods=['GET'])
def handle_feedback():
    fact_id = request.args.get('id')
    score = request.args.get('score')
    
    if fact_id and score:
        update_feedback(int(fact_id), int(score))
        # This HTML response shows up on your phone screen
        return f"""
        <html>
            <body style="font-family:sans-serif; text-align:center; padding-top:50px;">
                <h1>✅ Success!</h1>
                <p>Fact <b>{fact_id}</b> has been rated <b>{score}</b>.</p>
                <p>You can close this window now.</p>
            </body>
        </html>
        """, 200
    return "Error: Missing Data", 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)