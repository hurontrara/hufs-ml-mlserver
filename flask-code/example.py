from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

# 이미지 파일들의 정보를 저장하는 딕셔너리
images = {
    'image1': '/static/images/bread.jpg',
    'image2': '/static/images/yes.jpg',
    # 추가 이미지가 있다면 여기에 계속 추가
}


@app.route('/')
def index():
    return render_template('home.html', image_list=images)


@app.route('/bread')
def bread():
    bread_path = '/static/images/bread.jpg'
    return render_template('/image_html/bread.html', image_path=bread_path)


@app.route('/doom')
def doom():
    doom_path = '/static/negative/doom.jpg'
    return render_template('/image_html/doom.html', image_path=doom_path)


@app.route('/tears')
def tears():
    tears_path = '/static/negative/tears.jpg'
    return render_template('/image_html/tears.html', image_path=tears_path)


@app.route('/happy')
def happy():
    happy_path = '/static/positive/happy.jpg'
    return render_template('/image_html/happy.html', image_path=happy_path)


@app.route('/yes')
def yes():
    yes_path = '/static/positive/yes.jpg'
    return render_template('/image_html/yes.html', image_path=yes_path)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80, debug=True)


@app.route('/image/<image_id>')
def show_image(image_id):
    if image_id in images:
        image_path = images[image_id]
        return render_template('image.html', image_path=image_path)
    else:
        return 'Image not found', 404


@app.route('/image-server')
def image_server():
    # 이미지 서버로 리다이렉트
    return render_template('image_server.html')
