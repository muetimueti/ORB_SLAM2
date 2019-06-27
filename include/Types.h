#ifndef ORB_SLAM2_TYPES_H
#define ORB_SLAM2_TYPES_H


#include <iostream>

namespace knuff
{
struct Point
{
float x;
float y;

Point() : x(0), y(0) {}

Point(float _x, float _y) : x(_x), y(_y) {}

bool inline operator==(const Point &other) const
{
    return x == other.x && y == other.y;
}

template <typename T> inline
friend Point operator*(const T s, const Point& pt)
{
    return Point(pt.x*s, pt.y*s);
}

template <typename T> inline
friend void operator*=(Point& pt, const T s)
{
    pt.x*=s;
    pt.y*=s;
}

friend std::ostream& operator<<(std::ostream& os, const Point& pt)
{
    os << "[" << pt.x << "," << pt.y << "]";
    return os;
}
};
class KeyPoint
{
public:
    Point pt;  // Points coordinates (x,y) in image space
    float size;
    float angle;
    float response;
    int octave;

    KeyPoint() : pt(), size(0), angle(-1), response(0), octave(0) {}

    KeyPoint(Point _pt, float _size=0, float _angle=0, float _response=0, int _octave=0) :
            pt(_pt), size(_size), angle(_angle), response(_response), octave(_octave) {}

    KeyPoint(float _x, float _y, float _size=0, float _angle=-1, float _response=0, int _octave=0) :
            pt(_x, _y), size(_size), angle(_angle), response(_response), octave(_octave) {}

    bool operator==(const KeyPoint &other) const
    {
        return pt == other.pt && size == other.size && angle == other.angle && response == other.response &&
               octave == other.octave;
    }

    friend std::ostream& operator<<(std::ostream& os, const KeyPoint& kpt)
    {
        os << kpt.pt << ": size=" << kpt.size << ", angle=" << kpt.angle << ", response=" << kpt.response <<
           ", octave=" << kpt.octave;
        return os;
    }
};
} //namespace knuff

#endif //ORB_SLAM2_TYPES_H
