# app.py
from flask import Flask, request, jsonify
from processes import gen_port  # Import your portfolio generator function

app = Flask(__name__)  # Initialize the Flask app

# Define a route for portfolio generation
@app.route('/generate', methods=['POST'])
def generate():
    """
    This endpoint expects a JSON body with:
      - coins: a list of coin symbols (e.g. ["BTC", "ETH", "SOL"])
      - email: a user's email address (for future integration)
    It returns a JSON response with the portfolio report.
    """
    # Retrieve JSON data (or form data if necessary)
    data = request.get_json() or request.form
    
    # Extract the coin list and email from the input
    coins = data.get("coins")
    email = data.get("email")
    
    # Simple validation: both coins and email must be provided
    if not coins or not email:
        return jsonify({"error": "Missing coins or email"}), 400
    
    try:
        # Generate the portfolio report using your backend logic
        portfolio_report = gen_port(coins)
    except Exception as e:
        # Return an error response if something goes wrong
        return jsonify({"error": f"Portfolio generation failed: {str(e)}"}), 500
    
    # Optionally, log the submission for debugging (can be removed later)
    print("Received submission:", coins, email)
    
    # Return the portfolio report as a JSON response
    return jsonify(portfolio_report)

if __name__ == '__main__':
    # Run the Flask server in debug mode for local development
    app.run(debug=True)
