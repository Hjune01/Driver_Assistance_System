import cv2
import numpy as np
import time
import onnxruntime
import tflite_runtime.interpreter as tflite
from gpiozero import LED, PWMOutputDevice # Buzzer 대신 PWMOutputDevice 임포트

# --- 1. 전역 설정 변수 (Global Configuration Variables) ---
WEBCAM_INDEX = 0 # 웹캠 장치 인덱스. 보통 0이지만, 여러 개 연결 시 달라질 수 있습니다.
WEBCAM_WIDTH = 640 # 웹캠 해상도 너비
WEBCAM_HEIGHT = 480 # 웹캠 해상도 높이

# 낮/밤 구분 및 화질 보정 설정
BRIGHTNESS_THRESHOLD = 70 # 낮/밤 구분을 위한 밝기 임계값 (0-255). 이 값보다 낮으면 밤으로 간주.
CLAHE_CLIP_LIMIT = 2.0 # CLAHE의 대비 제한 (값이 클수록 대비 강화)
CLAHE_TILE_GRID_SIZE = (8, 8) # CLAHE 적용할 타일 그리드 크기
MEDIAN_BLUR_KERNEL = 3 # 미디언 블러 커널 크기 (홀수 값, 값이 클수록 노이즈 제거 강하지만 디테일 손실)
GAMMA_CORRECTION = 1.5 # 감마 값 (1.0보다 크면 밝아짐, 1.0보다 작으면 어두워짐)

# 차량 탐지 (ONNX) 설정
ONNX_MODEL_PATH = '/home/kjs/YOLO/project/best.onnx' # ONNX 모델 파일 경로 (수정됨!)
ONNX_INPUT_SIZE = 640 # 모델 입력 이미지 크기
ONNX_CLASS_NAMES = ['car'] # 클래스 이름
ONNX_CONF_THRESHOLD = 0.25 # 신뢰도 임계값
ONNX_IOU_THRESHOLD = 0.45 # NMS를 위한 IoU 임계값

# 차선 탐지 (TFLite) 설정
H5_MODEL_PATH = 'LLDNet.tflite' # TFLite 모델 파일 경로 (예: LLDNet.tflite)
H5_INPUT_WIDTH = 160 # TFLite 모델 입력 이미지 너비
H5_INPUT_HEIGHT = 80 # TFLite 모델 입력 이미지 높이
H5_RECENT_FIT_COUNT = 5 # 차선 예측 평균을 위한 이전 프레임 개수
LANE_THRESHOLD = 0.5 # 0-1 스케일의 모델 출력에 대한 차선 임계값 (이 값 이상이면 차선으로 간주)

# LED 및 부저 제어 설정
LED_PIN = 17 # LED에 연결된 라즈베리파이 GPIO 핀 번호 (BCM 모드 기준)
BUZZER_PIN = 18 # 부저에 연결된 라즈베리파이 GPIO 핀 번호 (BCM 모드 기준)
BUZZER_DISTANCE_THRESHOLD_CM = 20 # 부저가 울리는 거리 임계값 (cm)
# 패시브 부저를 위한 추가 설정
BUZZER_FREQUENCY_HZ = 1000 # 부저가 울릴 때의 주파수 (Hz) - '삐' 소리. 필요에 따라 조절
BUZZER_DUTY_CYCLE = 0.5 # 부저 듀티 사이클 (0.0 ~ 1.0) - 소리 크기 및 명확성 조절

# 거리 감지 설정 (⭐️ 중요: 실제 카메라 캘리브레이션 값으로 교체 필수!)
KNOWN_REAL_WIDTH = 1.8 # 미터 (일반적인 승용차의 평균 너비)
FOCAL_LENGTH_PX = 500.0 # ⭐️ 웹캠 캘리브레이션을 통해 얻은 실제 초점 거리(fx)로 교체하세요!

# --- 2. 낮/밤 구분 및 화질 보정 함수 (Day/Night Detection & Enhancement Functions) ---

def adjust_gamma(image, gamma=1.0):
    """감마 보정 적용 함수"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_night_frame(frame):
    """밤 이미지 화질 보정 함수"""
    # 1. BGR -> YUV (CLAHE는 Y 채널에 주로 적용)
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_frame)

    # 2. CLAHE 적용 (대비 강조)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    cl_y = clahe.apply(y)

    # 3. YUV 다시 병합 -> BGR
    merged_yuv = cv2.merge([cl_y, u, v])
    processed_frame = cv2.cvtColor(merged_yuv, cv2.COLOR_YUV2BGR)

    # 4. 미디언 블러 (노이즈 감소)
    processed_frame = cv2.medianBlur(processed_frame, MEDIAN_BLUR_KERNEL)

    # 5. 감마 보정 (전체적인 밝기 조절)
    processed_frame = adjust_gamma(processed_frame, GAMMA_CORRECTION)

    return processed_frame

def process_day_night_and_enhance(frame):
    """
    낮/밤을 구분하고 필요시 화질 보정을 적용합니다.
    반환값: (원본 프레임, 처리된 프레임, 현재 상태 텍스트, 텍스트 색상, 평균 밝기, 밤 여부)
    """
    original_frame = frame.copy() # 원본 프레임을 복사하여 보존
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_frame)

    if average_brightness < BRIGHTNESS_THRESHOLD:
        status_text = "Night (Processed)"
        processed_frame = enhance_night_frame(frame)
        text_color = (0, 0, 255) # 빨간색
        is_night = True
    else:
        status_text = "Day (Original)"
        processed_frame = frame.copy() # 원본 프레임 복사본 사용
        text_color = (0, 255, 0) # 초록색
        is_night = False
    
    return original_frame, processed_frame, status_text, text_color, average_brightness, is_night

# --- 3. 차량 탐지 클래스 (Vehicle Detection Class) ---

class VehicleDetector:
    def __init__(self, model_path, input_size, class_names, conf_threshold, iou_threshold, focal_length_px, known_real_width):
        self.model_path = model_path
        self.input_size = input_size
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.focal_length_px = focal_length_px
        self.known_real_width = known_real_width
        self.session = None
        self.input_name = None
        self.output_names = None
        self._load_model()

    def _load_model(self):
        """ONNX 모델을 로드합니다."""
        try:
            # CPUExecutionProvider는 라즈베리 파이에서 가장 일반적인 선택입니다.
            self.session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"ONNX 모델 '{self.model_path}' 로드 성공.")
            print(f"입력 이름: {self.input_name}, 출력 이름: {self.output_names}")
        except Exception as e:
            print(f"오류: ONNX 모델 로드 실패 - {e}")
            print("ONNX_MODEL_PATH가 올바른지, onnxruntime이 제대로 설치되었는지 확인하세요.")
            exit()

    def _estimate_distance(self, pixel_width):
        """
        탐지된 객체의 픽셀 너비를 기반으로 거리를 추정합니다.
        D = (W * F) / P
        D: 거리 (미터)
        W: 물체의 실제 너비 (미터)
        F: 카메라의 초점 거리 (픽셀)
        P: 이미지 상의 물체 픽셀 너비
        """
        if pixel_width == 0 or self.focal_length_px == 0: # 0으로 나누는 것을 방지
            return float('inf')
        
        # 기본 거리 추정 공식 (미터 단위)
        distance_m = (self.known_real_width * self.focal_length_px) / pixel_width
        
        # ⭐️ 사용자 요청에 따라 계산된 거리에 20을 나누고 100을 곱하여 cm로 변환 (임시 방편)
        distance_cm = (distance_m / 20.0) * 100.0
        
        return distance_cm

    def detect(self, frame):
        """
        주어진 프레임에서 차량을 탐지하고 바운딩 박스를 그립니다.
        반환값: (탐지된 차량이 그려진 프레임, 탐지된 차량 목록)
        """
        if self.session is None:
            print("오류: ONNX 모델이 로드되지 않았습니다.")
            return frame, []

        original_height, original_width = frame.shape[:2]

        # 이미지 전처리: 모델 입력 크기에 맞게 리사이즈 (640x640)
        resized_frame = cv2.resize(frame, (self.input_size, self.input_size))
        # BGR -> RGB 변환, HWC -> CHW 변환, 배치 차원 추가, 0-1 정규화
        input_tensor = resized_frame[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, HWC to CHW
        input_tensor = np.expand_dims(input_tensor, 0) # Add batch dimension (1, C, H, W)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0 # Normalize to 0-1

        # 추론 실행
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})[0]
        except Exception as e:
            print(f"ONNX 추론 중 오류 발생: {e}")
            return frame, []

        # 후처리 (바운딩 박스 추출 및 NMS)
        boxes = []
        confidences = []
        class_ids = []
        
        predictions = outputs[0] # 첫 번째 배치 (batch-size 1)

        for det in predictions:
            obj_conf = det[4] # 객체 존재 신뢰도
            
            if obj_conf > self.conf_threshold:
                scores = det[5:] # 클래스 스코어
                class_id = np.argmax(scores) # 가장 높은 스코어를 가진 클래스 ID
                class_conf = scores[class_id] # 해당 클래스의 신뢰도

                final_confidence = obj_conf * class_conf

                if final_confidence > self.conf_threshold:
                    # 바운딩 박스 좌표를 원본 이미지 스케일로 변환
                    center_x = det[0] * original_width / self.input_size
                    center_y = det[1] * original_height / self.input_size
                    width = det[2] * original_width / self.input_size
                    height = det[3] * original_height / self.input_size

                    # (x1, y1, width, height) 포맷으로 변환
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(final_confidence))
                    class_ids.append(class_id)

        # NMS (Non-Maximum Suppression) 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        detected_vehicles = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.class_names[class_ids[i]])
                confidence = confidences[i]
                
                # --- 거리 추정 로직 ---
                distance_cm = self._estimate_distance(w) # 바운딩 박스 너비(w)로 거리 추정
                
                detected_vehicles.append({
                    'box': (x, y, w, h), 
                    'label': label, 
                    'confidence': confidence,
                    'distance': distance_cm # 거리 정보 (cm) 추가
                })

                # 바운딩 박스 그리기
                box_color = (0, 255, 0) # 초록색 (BGR) - 박스 색상은 그대로 둠
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                # 텍스트 배경 그리기 (라벨 + 거리)
                text = f"{label}: {confidence:.2f} | Dist: {distance_cm:.2f}cm" # 단위 변경!
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # 텍스트가 위로 넘어가지 않도록 y 좌표 조정
                text_y = y - baseline
                if text_y < text_height: # 화면 상단에 너무 붙지 않도록 조정
                    text_y = y + text_height + baseline
                    cv2.rectangle(frame, (x, y), (x + text_width, y + text_height + baseline), box_color, -1)
                else:
                    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y), box_color, -1)
                
                # 텍스트 쓰기 - 여기를 빨간색으로 변경
                cv2.putText(frame, text, (x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # (B,G,R) = (0,0,255)는 빨간색
                
        return frame, detected_vehicles

# --- 4. 차선 탐지 클래스 (Lane Detection Class) - TFLite 기반 ---

class LaneDetector:
    def __init__(self, model_path, input_width, input_height, recent_fit_count, lane_threshold):
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.recent_fit_count = recent_fit_count
        self.lane_threshold = lane_threshold # 차선 임계값 추가
        self.interpreter = None # TFLite Interpreter 객체
        self.input_details = None
        self.output_details = None
        self.recent_fit = [] # 이전 예측들을 저장하여 평균을 계산 (차선 부드럽게 유지)
        self._load_model()

    def _load_model(self):
        """TFLite 모델을 로드합니다."""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"TFLite 모델 '{self.model_path}' 로드 성공.")
            print(f"입력 이름: {self.input_details[0]['name']}, 출력 이름: {self.output_details[0]['name']}")
            
            # 모델 출력의 채널 수를 확인하여 시각화 로직에 반영할 수 있습니다.
            # print(f"TFLite 모델 출력 형상: {self.output_details[0]['shape']}")

        except Exception as e:
            print(f"오류: TFLite 모델 로드 실패 - {e}")
            print("H5_MODEL_PATH가 .tflite 파일로 올바르게 변경되었는지, tflite-runtime이 제대로 설치되었는지 확인하세요.")
            exit()

    def detect(self, frame):
        """
        주어진 프레임에서 차선을 탐지하고 그립니다.
        반환값: 차선이 그려진 프레임
        """
        if self.interpreter is None:
            print("오류: TFLite 모델이 로드되지 않았습니다.")
            return frame

        # 전처리: TFLite 모델 입력 크기(160x80)에 맞게 리사이즈
        small_img = cv2.resize(frame, (self.input_width, self.input_height))
        
        # 모델 입력 타입 및 형식에 맞게 변환 (일반적으로 float32, 채널 마지막)
        input_data = np.array(small_img, dtype=np.float32)
        # TFLite 모델이 0-1 정규화를 요구한다면 여기서 / 255.0 을 해줘야 함
        # 예: input_data = np.array(small_img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0) # 배치 차원 추가

        # 모델 입력에 데이터 설정
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # 추론 실행
        self.interpreter.invoke()

        # 출력 데이터 가져오기
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # 첫 번째 배치 (batch-size 1)

        # ⭐️ 차선 시각화 로직 시작 ⭐️
        # 모델 출력이 0-1 사이의 확률값이라고 가정하고 임계값을 적용하여 이진 마스크 생성
        # LLDNet 모델의 출력에 따라 이 부분을 조정해야 합니다.
        _, binary_mask = cv2.threshold(prediction.astype(np.float32), self.lane_threshold, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8) # U8 타입으로 변환

        # 평균화 (부드러운 차선 유지를 위함)
        self.recent_fit.append(binary_mask) # 이진 마스크를 저장
        if len(self.recent_fit) > self.recent_fit_count:
            self.recent_fit = self.recent_fit[1:]
        
        # 평균을 낸 후 다시 이진화 (혹은 단순히 float로 평균낸 후 이진화)
        # 여기서는 평균 낸 후 다시 이진화하는 방식으로 진행
        avg_fit_float = np.mean(np.array(self.recent_fit), axis=0)
        _, avg_fit_binary = cv2.threshold(avg_fit_float.astype(np.float32), self.lane_threshold, 255, cv2.THRESH_BINARY)
        avg_fit = avg_fit_binary.astype(np.uint8)

        # avg_fit이 (H, W) 형태의 단일 채널 이미지인 경우:
        if len(avg_fit.shape) == 2: # 단일 채널인 경우
            blanks = np.zeros_like(avg_fit).astype(np.uint8)
            # 차선을 초록색으로 표시하기 위해 G 채널에만 값을 넣고 B, R 채널은 0으로 채움
            lane_drawn = np.dstack((blanks, avg_fit, blanks)) 
        # avg_fit이 이미 (H, W, 3) 형태의 3채널 이미지인 경우 (이 경우도 LLDNet이 직접 컬러 마스크를 줄 때)
        elif len(avg_fit.shape) == 3 and avg_fit.shape[2] == 3:
            lane_drawn = avg_fit 
        else:
            print(f"경고: 예상치 못한 TFLite 모델 출력 형상: {avg_fit.shape}. 차선 시각화에 문제가 있을 수 있습니다.")
            lane_drawn = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8) # 기본값으로 검은색 이미지 반환

        # ⭐️ 차선 시각화 로직 끝 ⭐️

        # 원본 프레임 크기에 맞춰 차선 마스크 리사이즈
        lane_image = cv2.resize(lane_drawn, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        # 원본 이미지와 차선 이미지 합성
        result_frame = cv2.addWeighted(frame, 1, lane_image, 1, 0)
        
        return result_frame

# --- 5. 메인 애플리케이션 실행 (Main Application Execution) ---
if __name__ == "__main__":
    # GPIO 설정 (gpiozero 사용)
    led = None # 초기화
    buzzer = None # 초기화
    try:
        led = LED(LED_PIN) # gpiozero.LED 객체 생성
        led.off() # 초기 LED 상태는 꺼짐
        print(f"GPIO {LED_PIN}번 핀 초기화 완료 (gpiozero).")
    except Exception as e:
        print(f"오류: GPIO {LED_PIN}번 핀 초기화 실패 - {e}")
        print("gpiozero 라이브러리가 설치되어 있는지, 올바른 핀 번호인지 확인하세요.")
        print("LED 제어 기능이 비활성화됩니다. 스크립트는 다른 기능을 계속 수행합니다.")

    try:
        # 부저를 Buzzer 대신 PWMOutputDevice로 초기화하여 주파수 제어 가능하도록 함
        buzzer = PWMOutputDevice(BUZZER_PIN, initial_value=0, frequency=BUZZER_FREQUENCY_HZ)
        print(f"GPIO {BUZZER_PIN}번 핀 (부저) 초기화 완료 (PWMOutputDevice).")
    except Exception as e:
        print(f"오류: GPIO {BUZZER_PIN}번 핀 (부저) 초기화 실패 - {e}")
        print("gpiozero 라이브러리가 설치되어 있는지, 올바른 핀 번호인지 확인하세요.")
        print("부저 제어 기능이 비활성화됩니다. 스크립트는 다른 기능을 계속 수행합니다.")
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    if not cap.isOpened():
        print(f"오류: 웹캠 (인덱스: {WEBCAM_INDEX})을 열 수 없습니다. 올바른 인덱스인지, 연결되어 있는지 확인하세요.")
        # GPIO가 초기화되었다면 종료 전 자원 해제
        if led is not None:
            led.close() # gpiozero LED 객체 닫기
        if buzzer is not None:
            buzzer.close() # gpiozero Buzzer 객체 닫기
        exit()

    print(f"웹캠 초기화 완료. 실제 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("모든 모델 로드 중...")

    # 차량 탐지기 초기화 (focal_length_px와 known_real_width 파라미터 전달)
    vehicle_detector = VehicleDetector(
        ONNX_MODEL_PATH, ONNX_INPUT_SIZE, ONNX_CLASS_NAMES, ONNX_CONF_THRESHOLD, ONNX_IOU_THRESHOLD,
        FOCAL_LENGTH_PX, KNOWN_REAL_WIDTH # 추가된 파라미터
    )

    # 차선 탐지기 초기화
    lane_detector = LaneDetector(
        H5_MODEL_PATH, H5_INPUT_WIDTH, H5_INPUT_HEIGHT, H5_RECENT_FIT_COUNT, LANE_THRESHOLD # 차선 임계값 전달
    )

    print("실시간 감지 및 보정 시작. 'q' 키를 눌러 종료하세요.")

    try:
        while True:
            ret, frame = cap.read() # 웹캠으로부터 프레임 읽기
            if not ret:
                print("오류: 웹캠에서 프레임을 읽어올 수 없습니다. 웹캠이 연결 해제되었거나 문제가 발생했습니다.")
                break

            processing_start_time = time.time()

            # 1. 낮/밤 구분 및 화질 보정
            original_raw_frame, processed_frame, day_night_status, text_color, brightness, is_night = process_day_night_and_enhance(frame)
            
            # 2. 차량 탐지
            # 차량 탐지는 보정된 프레임에 적용
            frame_with_vehicles, detected_vehicles = vehicle_detector.detect(processed_frame)

            # 3. 차선 탐지
            # 차선 탐지도 보정된 프레임에 적용 (차량 탐지 결과가 적용된 프레임 사용)
            final_display_frame = lane_detector.detect(frame_with_vehicles)

            # --- LED 제어 로직 (gpiozero 사용) ---
            if led is not None: # LED 객체가 성공적으로 생성된 경우에만 제어
                # 차량이 탐지되지 않은 밤에만 LED 켜기
                if is_night and not detected_vehicles:
                    led.on() # LED 켜기
                    cv2.putText(final_display_frame, "LED ON (Night & No Cars)", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA) # 노란색 텍스트
                else:
                    led.off() # LED 끄기
                    cv2.putText(final_display_frame, "LED OFF", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA) # 노란색 텍스트
            else: # LED 객체가 생성되지 않은 경우 (초기화 실패)
                cv2.putText(final_display_frame, "LED CONTROL DISABLED", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA) # 빨간색 텍스트

            # --- 부저 제어 로직 (gpiozero 사용) ---
            if buzzer is not None: # 부저 객체가 성공적으로 생성된 경우에만 제어
                # 가장 가까운 차량의 거리 확인
                min_distance = float('inf')
                if detected_vehicles:
                    min_distance = min([v['distance'] for v in detected_vehicles])
                
                if min_distance <= BUZZER_DISTANCE_THRESHOLD_CM:
                    # 패시브 부저를 울릴 때 주파수와 듀티 사이클 설정
                    buzzer.value = BUZZER_DUTY_CYCLE # 듀티 사이클 설정 (소리 켜짐)
                    buzzer.frequency = BUZZER_FREQUENCY_HZ # 주파수 설정
                    cv2.putText(final_display_frame, f"BUZZER ON! D:{min_distance:.2f}cm", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA) # 빨간색 텍스트
                else:
                    buzzer.value = 0 # 듀티 사이클 0으로 설정 (소리 꺼짐)
                    cv2.putText(final_display_frame, "BUZZER OFF", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA) # 초록색 텍스트
            else: # 부저 객체가 생성되지 않은 경우 (초기화 실패)
                cv2.putText(final_display_frame, "BUZZER CONTROL DISABLED", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA) # 빨간색 텍스트

            # 최종 프레임에 낮/밤 상태 및 밝기 표시
            cv2.putText(final_display_frame, f"Brightness: {brightness:.2f} ({day_night_status})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

            processing_end_time = time.time()
            overall_fps = 1 / (processing_end_time - processing_start_time)
            cv2.putText(final_display_frame, f"Overall FPS: {overall_fps:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA) # 노란색 FPS

            # --- 새로운 로직: 두 개의 창에 표시 ---
            # 원본 프레임에도 동일한 밝기/FPS 정보 표시 (선택 사항)
            cv2.putText(original_raw_frame, f"Brightness: {brightness:.2f} (Original)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(original_raw_frame, f"Overall FPS: {overall_fps:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(original_raw_frame, "ORIGINAL FEED", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 원본 피드 창
            cv2.imshow('Original Webcam Feed', original_raw_frame)
            # 처리된 피드 창
            cv2.imshow('Processed Vision System', final_display_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n프로그램 강제 종료.")
    finally:
        cap.release() # 웹캠 자원 해제
        cv2.destroyAllWindows() # 모든 OpenCV 창 닫기
        if led is not None:
            led.close() # gpiozero LED 객체 닫기
        if buzzer is not None:
            buzzer.close() # gpiozero Buzzer 객체 닫기
        print("통합 비전 시스템을 종료합니다.")
