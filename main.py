from Logica import Logica
import performance

if __name__ == '__main__':
    l = Logica(0.9, 0.2, 400)
    l.run()
    performance.performance()

