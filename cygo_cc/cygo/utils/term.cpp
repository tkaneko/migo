#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

int term_width(int default_width) {
    struct winsize ws;
    int fd;

    fd = open("/dev/tty", O_RDWR);
    if (fd < 0 || ioctl(fd, TIOCGWINSZ, &ws) < 0) return default_width;

    int ret = ws.ws_col;
    close(fd);

    return ret;
}
