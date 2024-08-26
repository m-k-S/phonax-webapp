class EventEmitter:
    def __init__(self):
        self.socketio = None

    def init_socketio(self, socketio):
        self.socketio = socketio

    def emit(self, event, data):
        if self.socketio:
            self.socketio.emit(event, data)
        else:
            print(f"SocketIO not initialized. Couldn't emit {event} with data {data}")

event_emitter = EventEmitter()