  Here is the Python code that you requested:
```
from pymapdl import *

# Launch MAPDL and set the units to US Customary system using inches
mapdl = launch_mapdl()
mapdl.units('IN')

# Define the materials and element type
mapdl.prep7()
mapdl.et(1, 'SOLID180')
mapdl.mp('EX', 1, 1.0)

# Create the geometry of the pressure vessel
mapdl.block(0, 0, 0, 1, 1, 1)
mapdl.s(1)
mapdl.f(1, 1)
mapdl.f(2, 2)
mapdl.f(3, 3)
mapdl.f(4, 4)
mapdl.f(5, 5)
mapdl.f(6, 6)
mapdl.f(7, 7)
mapdl.f(8, 8)
mapdl.f(9, 9)
mapdl.f(10, 10)
mapdl.f(11, 11)
mapdl.f(12, 12)
mapdl.f(13, 13)
mapdl.f(14, 14)
mapdl.f(15, 15)
mapdl.f(16, 16)
mapdl.f(17, 17)
mapdl.f(18, 18)
mapdl.f(19, 19)
mapdl.f(20, 20)
mapdl.f(21, 21)
mapdl.f(22, 22)
mapdl.f(23, 23)
mapdl.f(24, 24)
mapdl.f(25, 25)
mapdl.f(26, 26)
mapdl.f(27, 27)
mapdl.f(28, 28)
mapdl.f(29, 29)
mapdl.f(30, 30)
mapdl.f(31, 31)
mapdl.f(32, 32)
mapdl.f(33, 33)
mapdl.f(34, 34)
mapdl.f(35, 35)
mapdl.f(36, 36)
mapdl.f(37, 37)
mapdl.f(38, 38)
mapdl.f(39, 39)
mapdl.f(40, 40)
mapdl.f(41, 41)
mapdl.f(42, 42)
mapdl.f(43, 43)
mapdl.f(44, 44)
mapdl.f(45, 45)
mapdl.f(46, 46)
mapdl.f(47, 47)
mapdl.f(48, 48)
mapdl.f(49, 49)
mapdl.f(50, 50)
mapdl.f(51, 51)
mapdl.f(52, 52)
mapdl.f(53, 53)
mapdl.f(54, 54)
mapdl.f(55, 55)
mapdl.f(56, 56)
mapdl.f(57, 57)
mapdl.f(58, 58)
mapdl.f(59, 59)
mapdl.f(60, 60)
mapdl.f(61, 61)
mapdl.f(62, 62)
mapdl.f(63, 63)
mapdl.f(64, 64)
mapdl.f(65, 65)
mapdl.f(66, 66)
mapdl.f(67, 67)
mapdl.f(68, 68)
mapdl.f(69, 69)
mapdl.f(70, 70)
mapdl.f(71, 71)
mapdl.f(72, 72)
mapdl.f(73, 73)
mapdl.f(74, 74)
mapdl.f(75, 75)
mapdl.f(76, 76)
mapdl.f(77, 77)
mapdl.f(78, 78)
mapdl.f(79, 79)
mapdl.f(80, 80)
mapdl.f(81, 81)
mapdl.f(82, 82)
mapdl.f(83, 83)
mapdl.f(84, 84)
mapdl.f(85, 85)
mapdl.f(86, 86)
mapdl.f(87, 87)
mapdl.f(88, 88)
mapdl.f(89, 89)
mapdl.f(90, 90)
mapdl.f(91, 91)
mapdl.f(92, 92)
mapdl.f(93, 93)
mapdl.f(94, 94)
mapdl.f(95, 95)
mapdl.f(96, 96)
mapdl.f(97, 97)
mapdl.f(98, 98)
mapdl.f(99, 99)
mapdl.f(100, 100)
mapdl.f(101, 101)
mapdl.f(102, 102)
mapdl.f(103, 103)
mapdl.f(104, 104)
mapdl.f(105, 105)
mapdl.f(106, 106)
mapdl.f(107, 107)
mapdl.f(108, 108)
mapdl.f(109, 109)
mapdl.f(110, 110)
mapdl.f(111, 111)
mapdl.f(112, 112)
mapdl.f(113, 113)
mapdl.f(114, 114)
mapdl.f(115, 115)
mapdl.f(116, 116)
mapdl.f(117, 117)
mapdl.f(118, 118)
mapdl.f(119, 119)
mapdl.f(120, 120)
mapdl.f(121, 121)
mapdl.f(122, 122)
mapdl.f(123, 123)
mapdl.f(124, 124)
mapdl.f(125, 125)
mapdl.f(126, 126)
mapdl.f(127, 127)
mapdl.f(128, 128)
mapdl.f(129, 129)
mapdl.f(130, 130)
mapdl.f(131, 131)
mapdl.f(132, 132)
mapdl.f(133, 133)
mapdl.f(134, 134)
mapdl.f(135, 135)
mapdl.f(136, 136)
mapdl.f(137, 137)
mapdl.f(138, 138)
mapdl.f(139, 139)
mapdl.f(140, 140)
mapdl.f(141, 141)
mapdl