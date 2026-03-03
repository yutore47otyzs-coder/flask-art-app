import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

classes = [
    "claude-monet",
    "pablo-picasso",
    "pierre-auguste-renoir",
    "salvador-dali",
    "vincent-van-gogh",
]
image_size = 448

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])

app = Flask(__name__)


# submitボタンを押した際にエラーが出た場合上の行のコメントアウトを削除し、your_secret_key_hereに任意の文字列を指定し、再度アプリケーションを実行してください。
app.secret_key = "art_app"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model("./ArtModel.keras")  # 学習済みモデルをロード


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 受け取った画像を読み込み、np形式に変換
            img = image.load_img(
                filepath, color_mode="rgb", target_size=(image_size, image_size)
            )
            img = image.img_to_array(img)

            data = preprocess_input(np.array([img]))
            # 変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")


if __name__ == "__main__":
    # Renderではポート番号を環境変数から取得する必要があります
    # port = int(os.environ.get("PORT", 10000))
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
