from typing import Literal

from pydantic import BaseModel, Field


class NetworkTraffic(BaseModel):
    duration: float = Field(ge=0)
    protocol_type: Literal["tcp", "udp", "icmp"]
    service: str
    flag: str
    src_bytes: float = Field(ge=0)
    dst_bytes: float = Field(ge=0)
    land: int = Field(ge=0)
    wrong_fragment: int = Field(ge=0)
    urgent: int = Field(ge=0)
    hot: int = Field(ge=0)
    num_failed_logins: int = Field(ge=0)
    logged_in: int = Field(ge=0, le=1)
    num_compromised: int = Field(ge=0)
    root_shell: int = Field(ge=0, le=1)
    su_attempted: int = Field(ge=0)
    num_root: int = Field(ge=0)
    num_file_creations: int = Field(ge=0)
    num_shells: int = Field(ge=0)
    num_access_files: int = Field(ge=0)
    num_outbound_cmds: int = Field(ge=0)
    is_host_login: int = Field(ge=0, le=1)
    is_guest_login: int = Field(ge=0, le=1)
    count: int = Field(ge=0)
    srv_count: int = Field(ge=0)
    serror_rate: float = Field(ge=0, le=1)
    srv_serror_rate: float = Field(ge=0, le=1)
    rerror_rate: float = Field(ge=0, le=1)
    srv_rerror_rate: float = Field(ge=0, le=1)
    same_srv_rate: float = Field(ge=0, le=1)
    diff_srv_rate: float = Field(ge=0, le=1)
    srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_count: int = Field(ge=0)
    dst_host_srv_count: int = Field(ge=0)
    dst_host_same_srv_rate: float = Field(ge=0, le=1)
    dst_host_diff_srv_rate: float = Field(ge=0, le=1)
    dst_host_same_src_port_rate: float = Field(ge=0, le=1)
    dst_host_srv_diff_host_rate: float = Field(ge=0, le=1)
    dst_host_serror_rate: float = Field(ge=0, le=1)
    dst_host_srv_serror_rate: float = Field(ge=0, le=1)
    dst_host_rerror_rate: float = Field(ge=0, le=1)
    dst_host_srv_rerror_rate: float = Field(ge=0, le=1)


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    absolute_shap_value: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_features: list[SHAPFeature]