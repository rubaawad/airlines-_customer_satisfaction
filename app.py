from flask import Flask, request, render_template, redirect
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Redirect root URL to the predict endpoint
@app.route('/')
def index():
    return redirect('/predict') 

# Define the predict endpoint
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    print("in predict_datapoint")
    
    # If the request method is GET, render the form
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Pass empty form_data for initial rendering
    
    # If the request method is POST, process the form data
    else:
        print("call CustomData")
        
        # Extract form data using request.form.get
        age = float(request.form.get('Age'))
        flight_distance = float(request.form.get('Flight_distance'))
        inflight_wifi_service = float(request.form.get('Inflight_wifi_service'))
        departure_arrival_time_convenient = float(request.form.get('Departure_arrival_time_convenient'))
        ease_of_online_booking = float(request.form.get('Ease_of_online_booking'))
        gate_location = float(request.form.get('Gate_location'))
        food_and_drink = float(request.form.get('Food_and_drink'))
        online_boarding = float(request.form.get('Online_boarding'))
        seat_comfort = float(request.form.get('Seat_comfort'))
        inflight_entertainment = float(request.form.get('Inflight_entertainment'))
        on_board_service = float(request.form.get('On_board_service'))
        leg_room_service = float(request.form.get('Leg_room_service'))
        baggage_handling = float(request.form.get('Baggage_handling'))
        checkin_service = float(request.form.get('Checkin_service'))
        inflight_service = float(request.form.get('Inflight_service'))
        cleanliness = float(request.form.get('Cleanliness'))
        arrival_delay_in_minutes = float(request.form.get('Arrival_delay_in_minutes'))
        gender = request.form.get('Gender')
        customer_type = request.form.get('Customer_type')
        type_of_travel = request.form.get('Type_of_travel')
        class_type = request.form.get('Class')
        
        # Create CustomData object with form data
        data = CustomData(
            Age=age,
            Flight_Distance=flight_distance,
            Inflight_wifi_service=inflight_wifi_service,
            Departure_Arrival_time_convenient=departure_arrival_time_convenient,
            Ease_of_Online_booking=ease_of_online_booking,
            Gate_location=gate_location,
            Food_and_drink=food_and_drink,
            Online_boarding=online_boarding,
            Seat_comfort=seat_comfort,
            Inflight_entertainment=inflight_entertainment,
            On_board_service=on_board_service,
            Leg_room_service=leg_room_service,
            Baggage_handling=baggage_handling,
            Checkin_service=checkin_service,
            Inflight_service=inflight_service,
            Cleanliness=cleanliness,
            Arrival_Delay_in_Minutes=arrival_delay_in_minutes,
            Gender=gender,
            Customer_Type=customer_type,
            Type_of_Travel=type_of_travel,
            Class=class_type
        )
        
        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        print("Before Prediction")

        # Initialize predict pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # Make prediction
        results = predict_pipeline.predict(pred_df)
        print("after Prediction", results)
        
        # Translate prediction to human-readable format
        prediction_text = "The Customer is likely to be neutral or dissatisfied" if results[0] == 1 else "The Customer is likely to be satisfied"
        
        # Pass form data back to the template
        form_data = {
            'Age': age,
            'Flight Distance': flight_distance,
            'Inflight wifi service': inflight_wifi_service,
            'Departure/Arrival time convenient': departure_arrival_time_convenient,
            'Ease of Online booking': ease_of_online_booking,
            'Gate location': gate_location,
            'Food and drink': food_and_drink,
            'Online boarding': online_boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': inflight_entertainment,
            'On-board service': on_board_service,
            'Leg room service': leg_room_service,
            'Baggage handling': baggage_handling,
            'Checkin service': checkin_service,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness,
            'Arrival Delay in Minutes': arrival_delay_in_minutes,
            'Gender': gender,
            'Customer Type': customer_type,
            'Type of Travel': type_of_travel,
            'Class': class_type
        }

        return render_template('index.html', prediction=prediction_text, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)