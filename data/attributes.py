from enum import Enum


# Enum class for attributes included in Wider Attribute dataset only
class WiderAttributes(Enum):
    # { 0”：“男性”，“1”：“长发”，“2”：“太阳镜”“3”：“帽子”，“4”：“T-shirt”，“5”：“长袖”，“6”：“正装”,
    # “7”：“短裤”，“8”：“牛仔裤”“9”：“长裤”“10”：“裙子”，“11”：“面罩”，“12”：“标志”“13”：“条纹”}
    MALE = 0
    LONGHAIR = 1
    SUNGLASS = 2
    HAT = 3
    TSHIRT = 4
    LONGSLEEVE = 5
    FORMAL = 6
    SHORTS = 7
    JEANS = 8
    LONGPANTS = 9
    SKIRT = 10
    FACEMASK = 11
    LOGO = 12
    STRIPE = 13

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in WiderAttributes]

    @staticmethod
    def list_attributes(opt):
        out_rec = opt.specified_recognizable_attrs if opt.output_recognizable else []

        def fuc(ar):
            if str(ar) in out_rec:
                return Attribute(ar, AttributeType.BINARY, rec_trainable=True)
            else:
                return Attribute(ar, AttributeType.BINARY, rec_trainable=False)

        attrs_spc = filter(lambda x: str(x) in opt.specified_attrs,
                           [attr for attr in WiderAttributes])
        return list(map(fuc, attrs_spc))


class BerkeleyAttributes(Enum):
    # { 0”：“男性”，“1”：“长发”，“2”：“太阳镜”“3”：“帽子”，“4”：“T-shirt”，“5”：“长袖”,
    # “6”：“短裤”，“7”：“牛仔裤”“8”：“长裤”}
    MALE = 0
    LONGHAIR = 1
    SUNGLASS = 2
    HAT = 3
    TSHIRT = 4
    LONGSLEEVE = 5
    SHORTS = 6
    JEANS = 7
    LONGPANTS = 8

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in BerkeleyAttributes]

    @staticmethod
    def list_attributes(opt):
        out_rec = opt.specified_recognizable_attrs if opt.output_recognizable else []

        def fuc(ar):
            if str(ar) in out_rec:
                return Attribute(ar, AttributeType.BINARY, rec_trainable=True)
            else:
                return Attribute(ar, AttributeType.BINARY, rec_trainable=False)

        attrs_spc = filter(lambda x: str(x) in opt.specified_attrs,
                           [attr for attr in BerkeleyAttributes])
        return list(map(fuc, attrs_spc))


class ErisedAttributes(Enum):
    GENDER = 0
    AGE = 1
    AGE_GROUP = 2
    # DRESS = 0
    # GLASSES = 1
    # UNDERCUT = 2
    # GREASY = 3
    # PREGNANT = 4
    # AGE = 5
    # FIGURE = 6
    # HAIRCOLOR = 7
    # ALOPECIA = 8
    # TOTTOO = 9
    # CARRY_KIDS = 10
    # GENDER = 11

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in ErisedAttributes]


class AttributeType(Enum):
    BINARY = 0
    MULTICLASS = 1
    NUMERICAL = 2


class Attribute:
    def __init__(self, key, tp, rec_trainable=False):
        assert isinstance(key, Enum)
        assert isinstance(tp, AttributeType)
        self.key = key
        self.name = str(key)
        self.data_type = tp
        self.rec_trainable = rec_trainable

    # Merge two attribute of same key to make them compatible
    def merge(self, other_attribute):
        assert isinstance(other_attribute, Attribute)
        assert self.key == other_attribute.key
        assert self.data_type == other_attribute.data_type

        return Attribute(self.key, self.data_type, self.rec_trainable or other_attribute.rec_trainable)

    def __str__(self):
        return self.name

