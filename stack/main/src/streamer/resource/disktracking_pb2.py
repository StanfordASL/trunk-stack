# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: disktracking.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'disktracking.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12\x64isktracking.proto\x12\x0c\x64isktracking\",\n\x0c\x44iskPosition\x12\n\n\x02id\x18\x01 \x01(\t\x12\x10\n\x08position\x18\x02 \x03(\x02\"\xd5\x01\n\rDiskPositions\x12\x46\n\x0e\x64isk_positions\x18\x01 \x03(\x0b\x32..disktracking.DiskPositions.DiskPositionsEntry\x12\x15\n\risGripperOpen\x18\x02 \x01(\x08\x12\x13\n\x0bisRecording\x18\x03 \x01(\x08\x1aP\n\x12\x44iskPositionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12)\n\x05value\x18\x02 \x01(\x0b\x32\x1a.disktracking.DiskPosition:\x02\x38\x01\"\"\n\x0f\x44iskPositionAck\x12\x0f\n\x07message\x18\x01 \x01(\t2i\n\x13\x44iskTrackingService\x12R\n\x13StreamDiskPositions\x12\x1a.disktracking.DiskPosition\x1a\x1b.disktracking.DiskPositions\"\x00\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'disktracking_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_DISKPOSITIONS_DISKPOSITIONSENTRY']._loaded_options = None
  _globals['_DISKPOSITIONS_DISKPOSITIONSENTRY']._serialized_options = b'8\001'
  _globals['_DISKPOSITION']._serialized_start=36
  _globals['_DISKPOSITION']._serialized_end=80
  _globals['_DISKPOSITIONS']._serialized_start=83
  _globals['_DISKPOSITIONS']._serialized_end=296
  _globals['_DISKPOSITIONS_DISKPOSITIONSENTRY']._serialized_start=216
  _globals['_DISKPOSITIONS_DISKPOSITIONSENTRY']._serialized_end=296
  _globals['_DISKPOSITIONACK']._serialized_start=298
  _globals['_DISKPOSITIONACK']._serialized_end=332
  _globals['_DISKTRACKINGSERVICE']._serialized_start=334
  _globals['_DISKTRACKINGSERVICE']._serialized_end=439
# @@protoc_insertion_point(module_scope)
