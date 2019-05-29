from ewmh import EWMH
ewmh = EWMH()


def frame(client):
    frame = client
    while frame.query_tree().parent != ewmh.root:
        frame = frame.query_tree().parent
    return frame


for client in ewmh.getClientList():
    print(frame(client).get_geometry())
