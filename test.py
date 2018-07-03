import threading
import main as func
import time

pool = []

def upLoad():

    # url = "http://127.0.0.1:8080/CopybookEvaluation/image/copybook/001"
    # # req = request.Request(url=url, data=data, headers={}, method="POST")
    # # request.Request()
    # files = {'file': open('./img.jpg', 'rb')}
    # r = requests.post(url=url,files=files)
    # print( ":" + r.text)
    print("upload")
    func.main('./img.jpg')


    return


def main():
    """

    :return:
    """
    for x in range(10):
        print(x)
        th = threading.Thread(target=upLoad,args=())
        pool.append(th)
    for i in pool:
        print("loading")
        i.start()
        # keep thread
    for i in pool:
        print('keeping')
        i.join()
    return


if __name__ == '__main__':
    var = time.time()
    main()
    print(time.time()-var)


