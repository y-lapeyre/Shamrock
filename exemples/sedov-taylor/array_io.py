import numpy as np


def write_array(fname, xa):
    fp = open(fname, "w")
    for x in xa:
        s = "%lf\n" % x
        fp.write(s)
    fp.close()


def read_string_array(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    for i in range(len(fl)):
        fl[i] = fl[i].strip("\n")
    fp.close()
    return fl


def read_string_array_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    for i in range(len(fl) - 1):
        fl[i] = fl[i + 1].strip("\n")
    fp.close()
    return fl


def read_array(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    x = np.fromfile(fname, dtype=np.float32, sep="\n")
    fp.close()
    return x


def read_array_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    x = np.fromfile(fname, dtype=np.float32, sep="\n")
    fp.close()
    return x[1 : len(x) - 1]


def write_two_arrays(fname, xa, ya):
    fp = open(fname, "w")
    for i in range(len(xa)):
        s = "% 10e\t% 10e\n" % (xa[i], ya[i])
        fp.write(s)
    fp.close()


def read_two_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] = float(fl[i].split()[0])
        y[i] = float(fl[i].split()[1])
    return x, y


def read_two_arrays_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] = float(fl[i + 1].split()[0])
        y[i] = float(fl[i + 1].split()[1])
    return x, y


def write_three_arrays(fname, xa, ya, za):
    fp = open(fname, "w")
    for i in range(len(xa)):
        s = "% 10e\t% 10e\t% 10e\n" % (xa[i], ya[i], za[i])
        fp.write(s)
    fp.close()


def read_three_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        x[i] = float(fl[i].split()[0])
        y[i] = float(fl[i].split()[1])
        z[i] = float(fl[i].split()[2])
    return x, y, z


def read_three_arrays_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    for i in range(n):
        x[i] = float(fl[i + 1].split()[0])
        y[i] = float(fl[i + 1].split()[1])
        z[i] = float(fl[i + 1].split()[2])

    return x, y, z


def write_four_arrays(fname, a, b, c, d):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\n" % (a[i], b[i], c[i], d[i])
        fp.write(s)
    fp.close()


def read_four_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
    return a, b, c, d


def read_four_arrays_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i + 1].split()[0])
        b[i] = float(fl[i + 1].split()[1])
        c[i] = float(fl[i + 1].split()[2])
        d[i] = float(fl[i + 1].split()[3])
    return a, b, c, d


def write_five_arrays(fname, a, b, c, d, f):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (a[i], b[i], c[i], d[i], f[i])
        fp.write(s)
    fp.close()


def write_six_arrays(fname, a, b, c, d, f, g):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (a[i], b[i], c[i], d[i], f[i], g[i])
        fp.write(s)
    fp.close()


def write_five_arrays_nlines(fname, a, b, c, d, f):
    fp = open(fname, "w")
    s = "%s\n" % len(a)
    fp.write(s)
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (a[i], b[i], c[i], d[i], f[i])
        fp.write(s)
    fp.close()


def read_five_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
    return a, b, c, d, f


def read_six_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
    return a, b, c, d, f, g


def write_seven_arrays(fname, a, b, c, d, f, g, h):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (
            a[i],
            b[i],
            c[i],
            d[i],
            f[i],
            g[i],
            h[i],
        )
        fp.write(s)
    fp.close()


def write_eight_arrays(fname, a, b, c, d, f, g, h, j):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (
            a[i],
            b[i],
            c[i],
            d[i],
            f[i],
            g[i],
            h[i],
            j[i],
        )
        fp.write(s)
    fp.close()


def read_seven_arrays_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i + 1].split()[0])
        b[i] = float(fl[i + 1].split()[1])
        c[i] = float(fl[i + 1].split()[2])
        d[i] = float(fl[i + 1].split()[3])
        f[i] = float(fl[i + 1].split()[4])
        g[i] = float(fl[i + 1].split()[5])
        h[i] = float(fl[i + 1].split()[6])
    return a, b, c, d, f, g, h


def read_seven_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
    return a, b, c, d, f, g, h


def write_seven_arrays(fname, a, b, c, d, f, g, h):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (
            a[i],
            b[i],
            c[i],
            d[i],
            f[i],
            g[i],
            h[i],
        )
        fp.write(s)
    fp.close()


def write_ten_arrays(fname, a, b, c, d, f, g, h, k, l, m):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n" % (
            a[i],
            b[i],
            c[i],
            d[i],
            f[i],
            g[i],
            h[i],
            k[i],
            l[i],
            m[i],
        )
        fp.write(s)
    fp.close()


def read_eight_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
    return a, b, c, d, f, g, h, j


def read_nine_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
    return a, b, c, d, f, g, h, j, k


def read_ten_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
    return a, b, c, d, f, g, h, j, k, l


def read_twelve_arrays_nline(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i + 1].split()[0])
        b[i] = float(fl[i + 1].split()[1])
        c[i] = float(fl[i + 1].split()[2])
        d[i] = float(fl[i + 1].split()[3])
        f[i] = float(fl[i + 1].split()[4])
        g[i] = float(fl[i + 1].split()[5])
        h[i] = float(fl[i + 1].split()[6])
        j[i] = float(fl[i + 1].split()[7])
        k[i] = float(fl[i + 1].split()[8])
        l[i] = float(fl[i + 1].split()[9])
        m[i] = float(fl[i + 1].split()[10])
        p[i] = float(fl[i + 1].split()[11])
    return a, b, c, d, f, g, h, j, k, l, m, p


def read_thirteen_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    q = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        q[i] = float(fl[i].split()[12])

    return a, b, c, d, f, g, h, j, k, l, m, p, q


def read_fifteen_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac


def read_fifteen_arrays_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i + 1].split()[0])
        b[i] = float(fl[i + 1].split()[1])
        c[i] = float(fl[i + 1].split()[2])
        d[i] = float(fl[i + 1].split()[3])
        f[i] = float(fl[i + 1].split()[4])
        g[i] = float(fl[i + 1].split()[5])
        h[i] = float(fl[i + 1].split()[6])
        j[i] = float(fl[i + 1].split()[7])
        k[i] = float(fl[i + 1].split()[8])
        l[i] = float(fl[i + 1].split()[9])
        m[i] = float(fl[i + 1].split()[10])
        p[i] = float(fl[i + 1].split()[11])
        aa[i] = float(fl[i + 1].split()[12])
        ab[i] = float(fl[i + 1].split()[13])
        ac[i] = float(fl[i + 1].split()[14])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac


def write_fifteen_arrays(fname, a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = (
            "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n"
            % (
                a[i],
                b[i],
                c[i],
                d[i],
                f[i],
                g[i],
                h[i],
                j[i],
                k[i],
                l[i],
                m[i],
                p[i],
                aa[i],
                ab[i],
                ac[i],
            )
        )
        fp.write(s)
    fp.close()


def read_seventeen_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, af


def write_seventeen_arrays(fname, a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, ae):
    fp = open(fname, "w")
    for i in range(len(a)):
        s = (
            "% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\t% 10e\n"
            % (
                a[i],
                b[i],
                c[i],
                d[i],
                f[i],
                g[i],
                h[i],
                j[i],
                k[i],
                l[i],
                m[i],
                p[i],
                aa[i],
                ab[i],
                ac[i],
                ad[i],
                ae[i],
            )
        )
        fp.write(s)
    fp.close()


def read_twenty_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, af, ag, ah, aj


def read_twenty_one_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
        ak[i] = float(fl[i].split()[20])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, af, ag, ah, aj, ak


def read_twenty_three_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    al = np.zeros(n)
    am = np.zeros(n)
    for i in range(n):
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
        ak[i] = float(fl[i].split()[20])
        al[i] = float(fl[i].split()[21])
        am[i] = float(fl[i].split()[22])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, af, ag, ah, aj, ak, al, am


def read_twenty_three_arrays_nlines(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl) - 1
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    al = np.zeros(n)
    am = np.zeros(n)
    for i in range(n):
        # print(i)
        a[i] = float(fl[i + 1].split()[0])
        b[i] = float(fl[i + 1].split()[1])
        c[i] = float(fl[i + 1].split()[2])
        d[i] = float(fl[i + 1].split()[3])
        f[i] = float(fl[i + 1].split()[4])
        g[i] = float(fl[i + 1].split()[5])
        h[i] = float(fl[i + 1].split()[6])
        j[i] = float(fl[i + 1].split()[7])
        k[i] = float(fl[i + 1].split()[8])
        l[i] = float(fl[i + 1].split()[9])
        m[i] = float(fl[i + 1].split()[10])
        p[i] = float(fl[i + 1].split()[11])
        aa[i] = float(fl[i + 1].split()[12])
        ab[i] = float(fl[i + 1].split()[13])
        ac[i] = float(fl[i + 1].split()[14])
        ad[i] = float(fl[i + 1].split()[15])
        af[i] = float(fl[i + 1].split()[16])
        ag[i] = float(fl[i + 1].split()[17])
        ah[i] = float(fl[i + 1].split()[18])
        aj[i] = float(fl[i + 1].split()[19])
        ak[i] = float(fl[i + 1].split()[20])
        al[i] = float(fl[i + 1].split()[21])
        am[i] = float(fl[i + 1].split()[22])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, af, ag, ah, aj, ak, al, am


def read_twenty_five_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    al = np.zeros(n)
    am = np.zeros(n)
    ap = np.zeros(n)
    aq = np.zeros(n)
    for i in range(n):
        # print(i)
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
        ak[i] = float(fl[i].split()[20])
        al[i] = float(fl[i].split()[21])
        am[i] = float(fl[i].split()[22])
        ap[i] = float(fl[i].split()[23])
        aq[i] = float(fl[i].split()[24])
    return a, b, c, d, f, g, h, j, k, l, m, p, aa, ab, ac, ad, af, ag, ah, aj, ak, al, am, ap, aq


def read_twenty_six_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    al = np.zeros(n)
    am = np.zeros(n)
    ap = np.zeros(n)
    aq = np.zeros(n)
    ar = np.zeros(n)

    for i in range(n):
        # print(i)
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
        ak[i] = float(fl[i].split()[20])
        al[i] = float(fl[i].split()[21])
        am[i] = float(fl[i].split()[22])
        ap[i] = float(fl[i].split()[23])
        aq[i] = float(fl[i].split()[24])
        ar[i] = float(fl[i].split()[25])

    return (
        a,
        b,
        c,
        d,
        f,
        g,
        h,
        j,
        k,
        l,
        m,
        p,
        aa,
        ab,
        ac,
        ad,
        af,
        ag,
        ah,
        aj,
        ak,
        al,
        am,
        ap,
        aq,
        ar,
    )


def read_twenty_nine_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    al = np.zeros(n)
    am = np.zeros(n)
    ap = np.zeros(n)
    aq = np.zeros(n)
    ar = np.zeros(n)
    au = np.zeros(n)
    av = np.zeros(n)
    ax = np.zeros(n)

    for i in range(n):
        # print(i)
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
        ak[i] = float(fl[i].split()[20])
        al[i] = float(fl[i].split()[21])
        am[i] = float(fl[i].split()[22])
        ap[i] = float(fl[i].split()[23])
        aq[i] = float(fl[i].split()[24])
        ar[i] = float(fl[i].split()[25])
        au[i] = float(fl[i].split()[26])
        av[i] = float(fl[i].split()[27])
        ax[i] = float(fl[i].split()[28])
    return (
        a,
        b,
        c,
        d,
        f,
        g,
        h,
        j,
        k,
        l,
        m,
        p,
        aa,
        ab,
        ac,
        ad,
        af,
        ag,
        ah,
        aj,
        ak,
        al,
        am,
        ap,
        aq,
        ar,
        au,
        av,
        ax,
    )


def read_thirty_two_arrays(fname):
    fp = open(fname, "r")
    fl = fp.readlines()
    n = len(fl)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    f = np.zeros(n)
    g = np.zeros(n)
    h = np.zeros(n)
    j = np.zeros(n)
    k = np.zeros(n)
    l = np.zeros(n)
    m = np.zeros(n)
    p = np.zeros(n)
    aa = np.zeros(n)
    ab = np.zeros(n)
    ac = np.zeros(n)
    ad = np.zeros(n)
    af = np.zeros(n)
    ag = np.zeros(n)
    ah = np.zeros(n)
    aj = np.zeros(n)
    ak = np.zeros(n)
    al = np.zeros(n)
    am = np.zeros(n)
    ap = np.zeros(n)
    aq = np.zeros(n)
    ar = np.zeros(n)
    au = np.zeros(n)
    av = np.zeros(n)
    ax = np.zeros(n)
    br = np.zeros(n)
    bu = np.zeros(n)
    bv = np.zeros(n)

    for i in range(n):
        # print(i)
        a[i] = float(fl[i].split()[0])
        b[i] = float(fl[i].split()[1])
        c[i] = float(fl[i].split()[2])
        d[i] = float(fl[i].split()[3])
        f[i] = float(fl[i].split()[4])
        g[i] = float(fl[i].split()[5])
        h[i] = float(fl[i].split()[6])
        j[i] = float(fl[i].split()[7])
        k[i] = float(fl[i].split()[8])
        l[i] = float(fl[i].split()[9])
        m[i] = float(fl[i].split()[10])
        p[i] = float(fl[i].split()[11])
        aa[i] = float(fl[i].split()[12])
        ab[i] = float(fl[i].split()[13])
        ac[i] = float(fl[i].split()[14])
        ad[i] = float(fl[i].split()[15])
        af[i] = float(fl[i].split()[16])
        ag[i] = float(fl[i].split()[17])
        ah[i] = float(fl[i].split()[18])
        aj[i] = float(fl[i].split()[19])
        ak[i] = float(fl[i].split()[20])
        al[i] = float(fl[i].split()[21])
        am[i] = float(fl[i].split()[22])
        ap[i] = float(fl[i].split()[23])
        aq[i] = float(fl[i].split()[24])
        ar[i] = float(fl[i].split()[25])
        au[i] = float(fl[i].split()[26])
        av[i] = float(fl[i].split()[27])
        ax[i] = float(fl[i].split()[28])
        br[i] = float(fl[i].split()[29])
        bu[i] = float(fl[i].split()[30])
        bv[i] = float(fl[i].split()[31])
    return (
        a,
        b,
        c,
        d,
        f,
        g,
        h,
        j,
        k,
        l,
        m,
        p,
        aa,
        ab,
        ac,
        ad,
        af,
        ag,
        ah,
        aj,
        ak,
        al,
        am,
        ap,
        aq,
        ar,
        au,
        av,
        ax,
        br,
        bu,
        bv,
    )
