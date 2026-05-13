"""
Android 开发助手领域插件

实现 rag_framework.domain.base.DomainPlugin 接口，
提供 Android 领域专用的提示词、查询分类、术语映射等。
"""

from android_domain.plugin import AndroidDomainPlugin

__all__ = ["AndroidDomainPlugin"]
