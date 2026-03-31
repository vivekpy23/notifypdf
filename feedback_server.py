from flask import Flask, request
app = Flask(__name__)

from db_manager import update_feedback

@app.route('/feedback', methods=['POST'])
def feedback():
    # 1. Try to get JSON first (common for modern APIs)
    data = request.get_json(silent=True)
    
    # 2. If JSON is empty, check the standard Form data (what ntfy usually sends)
    if not data:
        data = request.form

    fid = data.get('id')
    val = data.get('score')
    
    if fid is not None and val is not None:
        update_feedback(int(fid), int(val))
        return "OK", 200
        
    return f"Invalid Data: {data}", 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)