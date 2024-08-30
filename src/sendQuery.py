import requests

def send_damage_data_with_image():
    # Prepare payload for the image
    data = {'status': 'damaged'}

    try:
        # POST request to send the status
        response = requests.post('http://localhost:3000/api/net', json=data)  # Use JSON payload
        print(response.status_code);
        if response.status_code == 200:
            print("Status sent successfully!")
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")

if __name__ == "__main__":
    send_damage_data_with_image()
