import cv2
import mediapipe as mp
import csv, random, math, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================== CẤU HÌNH ==================
WINDOW_W, WINDOW_H = 1760, 990
TIME_LIMIT = 30
PINCH_THRES = 0.05
CSV_FILE = "cauhoi.csv"

BASE_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.join(BASE_DIR, "Roboto", "Roboto-VariableFont_wdth,wght.ttf")

def get_font(size):
    return ImageFont.truetype(FONT_PATH, size)

# ================== VẼ CHỮ & HÌNH ==================
def draw_text_pil_bgr(img_bgr, text, xy, size=32, color=(255,255,255), anchor=None):
    try:
        font = get_font(size)
    except:
        # fallback nếu font thiếu
        cv2.putText(img_bgr, text, (int(xy[0]), int(xy[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, size/36.0, color, 2, cv2.LINE_AA)
        return img_bgr
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, font=font, fill=(color[2],color[1],color[0]), anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_text_in_box(img_bgr, text, box, size=28, color=(0,0,0), margin=12, center=False):
    x1,y1,x2,y2 = box
    try:
        font = get_font(size)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)

        max_w = x2-x1-2*margin
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = (cur+" "+w).strip()
            if draw.textlength(test, font=font) <= max_w or not cur:
                cur = test
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)

        line_h = (draw.textbbox((0,0), "Hg", font=font)[3] -
                  draw.textbbox((0,0), "Hg", font=font)[1]) + 6
        total_h = len(lines)*line_h - 6
        y = y1 + ((y2-y1)-total_h)//2 if center else y1+margin

        for ln in lines:
            draw.text((x1+margin,y), ln, font=font, fill=(color[2],color[1],color[0]))
            y += line_h

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img_bgr, text, (x1+margin, y1+40),
                    cv2.FONT_HERSHEY_SIMPLEX, size/36.0, color, 2, cv2.LINE_AA)
        return img_bgr

def rounded_rect(img, box, color, radius=16, thickness=-1):
    x1,y1,x2,y2 = box
    overlay = img.copy()
    cv2.rectangle(overlay,(x1+radius,y1),(x2-radius,y2),color,thickness)
    cv2.rectangle(overlay,(x1,y1+radius),(x2,y2-radius),color,thickness)
    for cx,cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                  (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(overlay,(cx,cy),radius,color,thickness)
    return cv2.addWeighted(overlay,1.0,img,0.0,0)

# ================== ĐỌC CSV (a;b;c + (đúng)) ==================
def load_questions_by_difficulty(filename):
    with open(filename,"r",encoding="utf-8") as f:
        rows = [row for row in csv.reader(f, delimiter=";") if row]
    if len(rows) < 2:
        raise ValueError("CSV không hợp lệ.")
    a,b,c = map(int, rows[0])
    easy_rows  = rows[1:1+a]
    med_rows   = rows[1+a:1+a+b]
    hard_rows  = rows[1+a+b:1+a+b+c]

    def parse(block, diff):
        out=[]
        for r in block:
            if len(r)<5: continue
            q = r[0].strip()
            ans = [s.strip() for s in r[1:5]]
            correct = 0
            for i,t in enumerate(ans):
                if "(đúng)" in t:
                    correct = i
                    ans[i] = t.replace("(đúng)","").strip()
            out.append({"question":q,"answers":ans,"correct":correct,"difficulty":diff})
        return out

    easy=parse(easy_rows,"easy"); med=parse(med_rows,"medium"); hard=parse(hard_rows,"hard")
    sel=[]
    sel += random.sample(easy, min(2,len(easy)))
    sel += random.sample(med,  min(2,len(med)))
    sel += random.sample(hard, min(1,len(hard)))
    random.shuffle(sel)
    return sel

def score_for(diff): return {"easy":1,"medium":2,"hard":4}[diff]

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def choose_right_hand(results):
    if not (results.multi_hand_landmarks and results.multi_handedness): return None
    for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        if handed.classification[0].label == "Right": return lm
    return results.multi_hand_landmarks[0]

def pinch_info(hand):
    a = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    b = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return math.dist((a.x,a.y),(b.x,b.y)), b

def inside(px,py,box):
    x1,y1,x2,y2 = box
    return x1<=px<=x2 and y1<=py<=y2

# ================== GAME ==================
def main():
    cv2.namedWindow("Quiz Game", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Quiz Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

    state = "waiting_player"
    total_score = 0
    questions = []
    q_idx = 0
    chosen_idx = None
    question_start = time.time()

    BTN_START  = (520, 420, 760, 500)
    BTN_NEXT   = (520, 620, 760, 700)
    BTN_RETRY  = (420, 420, 640, 500)
    BTN_CHANGE = (660, 420, 880, 500)

    with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.7) as hands:
        last_pinch = False

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame,1)
            h,w,_ = frame.shape

            header_box = (40, 20, w-40, 120)
            timer_box_outer = (w-300, 20, w-40, 60)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            is_pinch=False; px=py=None

            hand = choose_right_hand(results) if results else None
            hand_draw_info = None
            hand_landmarks = None
            if hand:
                dist, tip = pinch_info(hand)
                thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)
                xm, ym = (x1 + x2) // 2, (y1 + y2) // 2
                hand_draw_info = (x1, y1, x2, y2, xm, ym, dist)
                hand_landmarks = hand
                if dist < PINCH_THRES:
                    is_pinch=True
                    px,py = int(tip.x*w), int(tip.y*h)

            # ========= WAITING PLAYER =========
            if state == "waiting_player":
                welcome_box = (380, 260, 900, 360)
                frame = rounded_rect(frame, welcome_box, (40,40,40), 18, -1)
                welcome_text = "Chào mừng bạn đến với 'Techie Quiz'!\nHãy nhấn nút BẮT ĐẦU để chơi."
                frame = draw_text_in_box(frame, welcome_text, welcome_box, 28, (255,255,255), center=True)

                frame = rounded_rect(frame, BTN_START, (0,180,0), 18, -1)
                frame = draw_text_in_box(frame, "BẮT ĐẦU", BTN_START, 32, (255,255,255), center=True)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: break

                if is_pinch and not last_pinch and px is not None:
                    if inside(px,py,BTN_START):
                        questions = load_questions_by_difficulty(CSV_FILE)
                        total_score = 0; q_idx = 0; chosen_idx = None
                        question_start = time.time()
                        state = "playing"

            # ========= PLAYING =========
            elif state == "playing":
                q = questions[q_idx]

                # nội dung câu hỏi
                header_box = (40, 80, w-40, 180)
                frame = rounded_rect(frame, header_box, (30,30,30), 16, -1)
                frame = draw_text_in_box(frame, q["question"], header_box, 32, (255,255,255))

                # đếm giờ cho mỗi câu
                elapsed = int(time.time() - question_start)
                remain = max(0, TIME_LIMIT - elapsed)
                frame = rounded_rect(frame, timer_box_outer, (0,0,0), 12, -1)
                full_w = timer_box_outer[2]-timer_box_outer[0]-4
                filled = int(full_w * (remain/TIME_LIMIT))
                prog_box = (timer_box_outer[0]+2, timer_box_outer[1]+2, timer_box_outer[0]+2+filled, timer_box_outer[3]-2)
                frame = rounded_rect(frame, prog_box, (0,180,255), 10, -1)
                # Thêm nền cho số giây
                sec_box = (timer_box_outer[2]-80, timer_box_outer[1], timer_box_outer[2]-10, timer_box_outer[3])
                frame = rounded_rect(frame, sec_box, (30,30,30), 8, -1)
                frame = draw_text_pil_bgr(frame, f"{remain}s", (timer_box_outer[2]-60, timer_box_outer[1]+6), 26)

                # đáp án
                ans_boxes=[]; sensor_boxes=[]
                left_x, right_x = 80, w-80-480
                top_y = 260; gap_y = 140; ans_w, ans_h = 480, 80
                sensor_w = 70
                for r in range(2):
                    y = top_y + r*gap_y
                    ans_boxes.append((left_x, y, left_x+ans_w, y+ans_h))
                    sensor_boxes.append((left_x+ans_w-sensor_w, y, left_x+ans_w, y+ans_h))
                    ans_boxes.append((right_x, y, right_x+ans_w, y+ans_h))
                    sensor_boxes.append((right_x+ans_w-sensor_w, y, right_x+ans_w, y+ans_h))

                for i, box in enumerate(ans_boxes):
                    frame = rounded_rect(frame, box, (210,210,210), 18, -1)
                    frame = draw_text_in_box(frame, f"{chr(65+i)}. {q['answers'][i]}", box, 28, (0,0,0), center=True)

                # vẽ vùng sensor box
                for sb in sensor_boxes:
                    frame = rounded_rect(frame, sb, (180,180,180), 18, -1)

                # Xử lý chọn đáp án qua vùng cảm biến
                if is_pinch and not last_pinch and px is not None:
                    for i, sb in enumerate(sensor_boxes):
                        if inside(px,py,sb):
                            chosen_idx = i
                            if i == q["correct"]:
                                total_score += score_for(q["difficulty"])
                            state = "review"
                            break

                # Nếu hết giờ thì chuyển sang review và không cộng điểm
                if remain == 0:
                    chosen_idx = None
                    state = "review"
            # ========= REVIEW =========
            elif state == "review":
                q = questions[q_idx]

                header_box = (40, 80, w-40, 180)
                frame = rounded_rect(frame, header_box, (30,30,30), 16, -1)
                frame = draw_text_in_box(frame, q["question"], header_box, 32, (255,255,255))

                ans_boxes=[]
                left_x, right_x = 80, w-80-480
                top_y = 260; gap_y = 140; ans_w, ans_h = 480, 80
                for r in range(2):
                    y = top_y + r*gap_y
                    ans_boxes.append((left_x, y, left_x+ans_w, y+ans_h))
                    ans_boxes.append((right_x, y, right_x+ans_w, y+ans_h))

                for i, box in enumerate(ans_boxes):
                    if i == q["correct"]:
                        color = (0,200,0)
                    elif i == chosen_idx:
                        color = (0,0,220)
                    else:
                        color = (210,210,210)
                    frame = rounded_rect(frame, box, color, 18, -1)
                    frame = draw_text_in_box(frame, f"{chr(65+i)}. {q['answers'][i]}", box, 28, (255,255,255) if i in (q["correct"],chosen_idx) else (0,0,0), center=True)

                # thông báo
                msg_box = (200, 160, w-200, 240)
                frame = rounded_rect(frame, msg_box, (50,50,50), 18, -1)
                # Nếu chosen_idx là None (hết giờ), hiển thị đáp án đúng
                if chosen_idx is None:
                    msg = f"Hết giờ! Đáp án đúng là: {chr(65+q['correct'])}. {q['answers'][q['correct']]}"
                elif chosen_idx == q["correct"]:
                    msg = f"Chúc mừng bạn đã giành được {score_for(q['difficulty'])} điểm!"
                else:
                    msg = "Thật tiếc khi lựa chọn vừa rồi là không đúng!"
                frame = draw_text_in_box(frame, msg, msg_box, 30, (255,255,255), center=True)

                # nút tiếp tục
                frame = rounded_rect(frame, BTN_NEXT, (0,160,0), 18, -1)
                frame = draw_text_in_box(frame, "TIẾP TỤC", BTN_NEXT, 30, (255,255,255), center=True)

                if is_pinch and not last_pinch and px is not None:
                    if inside(px,py,BTN_NEXT):
                        q_idx += 1
                        if q_idx >= len(questions):
                            state = "game_over"
                        else:
                            chosen_idx = None
                            question_start = time.time()
                            state = "playing"

            # ========= GAME OVER =========
            elif state == "game_over":
                overlay = frame.copy()
                cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),-1)
                frame = cv2.addWeighted(overlay,0.5,frame,0.5,0)
                msg = f"Bạn đã giành được {total_score} điểm!"
                frame = draw_text_pil_bgr(frame, msg, (w//2, h//2-60), 44, (0,255,255), anchor="ms")

                frame = rounded_rect(frame, BTN_RETRY, (0,160,0), 18, -1)
                frame = draw_text_in_box(frame, "KẾT THÚC", BTN_RETRY, 30, (255,255,255), center=True)

                if is_pinch and not last_pinch and px is not None:
                    if inside(px,py,BTN_RETRY):
                        total_score = 0; q_idx = 0; chosen_idx = None
                        state = "waiting_player"


            # nhắc về thao tác chụm ngón tay
            note_box = (w//2-220, h-38, w//2+220, h-10)
            frame = rounded_rect(frame, note_box, (0,0,0), 8, -1)
            frame = draw_text_in_box(frame, "Chụm ngón tay cái và trỏ = nhấn nút", note_box, 20, (255, 255, 255), center=True)

            last_pinch = is_pinch
            # ===== Ô điểm (chỉ hiện khi đang chơi hoặc review) =====
            if state in ("playing", "review"):
                score_box = (w-260, h-80, w-40, h-40)
                frame = rounded_rect(frame, score_box, (0,0,0), 12, -1)
                frame = draw_text_in_box(frame, f"Điểm: {total_score}", score_box, 28, (0,255,255), center=True)
            # VẼ KHUNG BÀN TAY & CHẤM NHẬN DẠNG TAY LÊN TRÊN CÙNG
            if hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if hand_draw_info:
                x1, y1, x2, y2, xm, ym, dist = hand_draw_info
                if dist < PINCH_THRES:
                    # Chụm: vẽ 1 chấm xanh lá ở giữa
                    cv2.circle(frame, (xm, ym), 22, (0,255,0), -1)
                else:
                    # Không chụm: vẽ 2 chấm đỏ ở hai đầu ngón
                    cv2.circle(frame, (x1, y1), 18, (0,0,255), -1)
                    cv2.circle(frame, (x2, y2), 18, (0,0,255), -1)

            cv2.imshow("Quiz Game", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()