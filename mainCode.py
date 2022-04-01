try:
    import cv2
except Exception as e:
    print("Uyarı: OpenCV yüklü değil. Hareket algılama özelliğini kullanmak için OpenCV'yi doğru yapılandırdığınızdan emin olun.")

import time
from thread import *          # eğer raspberry de çalışmazsa:  from thread import *      ya da  from threading import Thread yap // import thread
import threading
import atexit
import sys
import termios          # eğer çaılşmazsa:   from termios import *    dene
import contextlib

import imutils
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor


### Kullanıcı Parametreleri ###

MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False

MAX_STEPS_X = 30
MAX_STEPS_Y = 15

RELAY_PIN = 22

###############################


@contextlib.contextmanager
def raw_mode(file):
    """
    Tuşlara basılmayı sağlayan fonksiyon.
    :param file:
    :return:
    """
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    # termios modülü ile ilgili bigi için bakın:   https://docs.python.org/3.4/library/termios.html
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


class VideoAraclar(object):   # Video Araçları sınıfı
    """
    Video araçları için yapıcı fonksiyon
    """
    @staticmethod
    def canli_video(camera_port=0):
        """
        Bir tane canlı video için pencere açıyor.
        :param camera:
        :return:
        """

        video_capture = cv2.VideoCapture(camera_port)

        while True:
            # çerçeve çerçeve(kare kare) yakalama.
            ret, frame = video_capture.read()

            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # herşey bittiğinde yakalamayı bırak
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def hareketi_bul(callback, camera_port=0, show_video=False):

        camera = cv2.VideoCapture(camera_port)
        time.sleep(0.25)

        # video akışındaki ilk kareyi başlatır.
        firstFrame = None
        tempFrame = None
        count = 0

        # video çerçevesi üzerinden döngü
        while True:
            # geçerli çerçeveyi kapatıp meşgul ol/meşgul olma
            # text

            (grabbed, frame) = camera.read()

            # çerçeve yakalanmazsa, videonun sonuna ulaştık.
            if not grabbed:
                break

            # çerçeveyi yeniden boyutlandır, gri ölçeğe çevir ve bulanıklaştır
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # ilk çerçeve yok ise onu başlat
            if firstFrame is None:
                print("Videonun ayarlanması bekleniyor...")
                if tempFrame is None:
                    tempFrame = gray
                    continue
                else:
                    delta = cv2.absdiff(tempFrame, gray)
                    tempFrame = gray
                    tst = cv2.threshold(delta, 5, 255, cv2.THRESH_BINARY)[1]
                    tst = cv2.dilate(tst, None, iterations=2)
                    if count > 30:
                        print("Tamamlanmış.\n Hareket bekleniyor.")
                        if not cv2.countNonZero(tst) > 0:
                            firstFrame = gray
                        else:
                            continue
                    else:
                        count += 1
                        continue

            # geçerli çerçeve ve ilk çerçeve arasındaki mutlak farkı hesapla.
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # eşik görüntüyü delikleri doldurmak için genişletir. Sonra eşitli görüntünün çevrelerini bul.
            thresh = cv2.dilate(thresh, None, iterations=2)
            c = VideoAraclar.get_best_contour(thresh.copy(), 5000)

            if c is not None:
                # çevre için sınırlayıcı kutuyu hesaplayın, çerçeveyi çiz ve metni güncelle.
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                callback(c, frame)

            # kullanıcı bir tuşa basarsa çerçeveyi gösterin ve kaydedin.
            if show_video:
                cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF

                # eğer q tuşuna basılırsa döngüden çık
                if key == ord("q"):
                    break

        # kamerayı temizle ve açık pencereyi kapat.
        camera.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_best_contour(imgmask, threshold):
        im, contours, hierarchy = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt


class Lazer(object):
    """
    Lazeri kullanmak için bu sınıf kullanılır.
    """
    def __init__(self, friendly_mode=True):
        self.friendly_mode = friendly_mode

        # varsayılan nesne oluştur, i2c adresinde ve frekansında değişiklik yok
        self.mh = Adafruit_MotorHAT()
        atexit.register(self.__turn_off_motors)

        # step motor 1
        self.sm_x = self.mh.getStepper(200, 1)      # 200 adım/devir, motor port 1
        self.sm_x.setSpeed(5)                       # 5 RPM
        self.current_x_steps = 0

        # step motor 2
        self.sm_y = self.mh.getStepper(200, 2)      # 200 adım/devir, motor port 2
        self.sm_y.setSpeed(5)                       # 5 RPM
        self.current_y_steps = 0

        # Röle
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    def calibrate(self):
        """
        girişin lazer eksenini kalibre etmesini bekler
        :return:
        """
        print("Lütfen lazerin yatay olabilmesi için eğimini kalibre edin. "
              "Komutlar: (w) yukarı hareket eder, (s) aşağı iner. İşlemi bitirmek için (enter) tuşuna basın.\n")
        self.__calibrate_y_axis()

        print("Lütfen lazerin rotasını kamerayla hizalanacak şekilde kalibre edin. "
              "Komutlar: (a) sola hareket eder, (d) sağa doğru hareket eder. İşlemi bitirmek için (enter) tuşuna basın.\n")
        self.__calibrate_x_axis()

        print("Kalibrasyon tamamlandı.")

    def __calibrate_x_axis(self):
        """
        x eksen kalibrasyanu için giriş bekle
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break

                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Lazer.move_backward(self.sm_x, 5)
                        else:
                            Lazer.move_forward(self.sm_x, 5)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Lazer.move_forward(self.sm_x, 5)
                        else:
                            Lazer.move_backward(self.sm_x, 5)
                    elif ch == "\n":
                        break

            except (KeyboardInterrupt, EOFError):
                print("Hata: Lazer kalibre edilemedi. Çıkılıyor...")
                sys.exit(1)

    def __calibrate_y_axis(self):
        """
        y ekseni kalibrasyonu için giriş bekle
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break

                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Lazer.move_forward(self.sm_y, 5)
                        else:
                            Lazer.move_backward(self.sm_y, 5)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Lazer.move_backward(self.sm_y, 5)
                        else:
                            Lazer.move_forward(self.sm_y, 5)
                    elif ch == "\n":
                        break

            except (KeyboardInterrupt, EOFError):
                print("Hata: Lazer kalibre edilemedi. Çıkılıyor...")
                sys.exit(1)

    def hareket_algilama(self, show_video=False):       # motion detection
        """
        Lazeri taşımak için kamerayı kullanır. OpenCV bunu kullanmak için yapılandırılmalıdır.
        :return:
        """
        VideoAraclar.hareketi_bul(self.__move_axis, show_video=show_video)

    def __move_axis(self, contour, frame):
        (v_h, v_w) = frame.shape[:2]
        (x, y, w, h) = cv2.boundingRect(contour)

        # yüksekliği bul
        target_steps_x = (2*MAX_STEPS_X * (x + w / 2) / v_w) - MAX_STEPS_X
        target_steps_y = (2*MAX_STEPS_Y*(y+h/2) / v_h) - MAX_STEPS_Y

        print("x: %s, y: %s" % (str(target_steps_x), str(target_steps_y)))
        print("Anlık x: %s, anlık y: %s" % (str(self.current_x_steps), str(self.current_y_steps)))

        t_x = threading.Thread()
        t_y = threading.Thread()
        t_fire = threading.Thread()

        # x ekseninde hareket
        if (target_steps_x - self.current_x_steps) > 0:
            self.current_x_steps += 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=Lazer.move_forward, args=(self.sm_x, 2,))
            else:
                t_x = threading.Thread(target=Lazer.move_backward, args=(self.sm_x, 2,))
        elif (target_steps_x - self.current_x_steps) < 0:
            self.current_x_steps -= 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=Lazer.move_backward, args=(self.sm_x, 2,))
            else:
                t_x = threading.Thread(target=Lazer.move_forward, args=(self.sm_x, 2,))

        # y ekseninde hareket
        if (target_steps_y - self.current_y_steps) > 0:
            self.current_y_steps += 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=Lazer.move_backward, args=(self.sm_y, 2,))
            else:
                t_y = threading.Thread(target=Lazer.move_forward, args=(self.sm_y, 2,))
        elif (target_steps_y - self.current_y_steps) < 0:
            self.current_y_steps -= 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=Lazer.move_forward, args=(self.sm_y, 2,))
            else:
                t_y = threading.Thread(target=Lazer.move_backward, args=(self.sm_y, 2,))

        # gerekirse tetikle
        if not self.friendly_mode:
            if abs(target_steps_y - self.current_y_steps) <= 2 and abs(target_steps_x - self.current_x_steps) <= 2:
                t_fire = threading.Thread(target=Lazer.fire)

        t_x.start()
        t_y.start()
        t_fire.start()

        t_x.join()
        t_y.join()
        t_fire.join()

    def interactive(self):
        """
        Etkileşimli bir oturum başlatır. Tuş vuruşları hareketi belirler. // tuşlar ile hareketli kontrol
        :return:
        """

        Lazer.move_forward(self.sm_x, 1)
        Lazer.move_forward(self.sm_y, 1)

        print('Komutlar: X ekseninde döndürme için (a) ve (d). Eğmek için (w) ve (s). Çıkış için (q)\n')
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break

                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Lazer.move_forward(self.sm_y, 5)
                        else:
                            Lazer.move_backward(self.sm_y, 5)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Lazer.move_backward(self.sm_y, 5)
                        else:
                            Lazer.move_forward(self.sm_y, 5)
                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Lazer.move_backward(self.sm_x, 5)
                        else:
                            Lazer.move_forward(self.sm_x, 5)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Lazer.move_forward(self.sm_x, 5)
                        else:
                            Lazer.move_backward(self.sm_x, 5)
                    elif ch == "\n":
                        Lazer.fire()

            except (KeyboardInterrupt, EOFError):
                pass

    @staticmethod
    def fire():
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    @staticmethod
    def move_forward(motor, steps):
        """
        Adım motorunu belirtilen sayıda adım ileriye doğru hareket ettirir.
        :param motor:
        :param steps:
        :return:
        """
        motor.step(steps, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.INTERLEAVE)

    @staticmethod
    def move_backward(motor, steps):
        """
        Step motorunu belirtilen adım sayısına geriye doğru hareket ettirir.
        :param motor:
        :param steps:
        :return:
        """
        motor.step(steps, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.INTERLEAVE)

    def __turn_off_motors(self):
        """
        Kapanma sırasında otomatik devre dışı bırakma motorları için önerilir!
        :return:
        """
        self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

if __name__ == "__main__":
    t = Lazer(friendly_mode=False)

    user_input = input("Giriş modunu seçin: (1) Hareket Algılayıcı, (2) İnteraktif\n")

    if user_input == "1":
        t.calibrate()
        if input("Canlı Video? (e, h)\n").lower() == "e":
            t.hareket_algilama(show_video=True)
        else:
            t.hareket_algilama()
    elif user_input == "2":
        if input("Canlı Video? (e, h)\n").lower() == "e":
            thread.start_new_thread(VideoAraclar.canli_video())
        t.interactive()
    else:
        print("Giriş seçeneği bilinmiyor. Lütfen (1) veya (2) tuşlarını seçin")