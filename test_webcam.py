from ultralytics import YOLO
import cv2
import argparse

# Load your trained model
model = YOLO('best.pt')  # Put best.pt in same folder

def run_webcam(conf_threshold=0.7):
    """Test model with webcam feed"""
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    # TRY THESE SETTINGS:
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Adjust brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Adjust contrast
    print("🎥 Webcam started. Press 'q' to quit.")
    print(f"📊 Confidence threshold: {conf_threshold}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model(frame, conf=conf_threshold)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Show detections in console
        if results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"  Detected: {name} ({conf:.1%})", end='\r')
        
        # Display video
        cv2.imshow('Currency Detector', annotated_frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam test complete!")

def test_image(image_path, conf_threshold=0.5):
    """Test on single image"""
    results = model(image_path, conf=conf_threshold)
    
    # Save and show result
    results[0].save(filename='test_result.jpg')
    print(f"✅ Result saved to 'test_result.jpg'")
    
    # Show in window
    cv2.imshow('Result', cv2.imread('test_result.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='webcam', 
                       choices=['webcam', 'image'], help='Test mode')
    parser.add_argument('--image', type=str, default='test.jpg', 
                       help='Image path for image mode')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        run_webcam(args.conf)
    else:
        test_image(args.image, args.conf)