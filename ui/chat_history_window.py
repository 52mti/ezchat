from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from services.chat_history_store import ChatHistoryStore


class ChatHistoryDialog(QDialog):
    def __init__(
        self,
        *,
        store: ChatHistoryStore,
        device_ip: str,
        app_name: str,
        app_label: str,
        persona: dict[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.device_ip = device_ip
        self.app_name = app_name
        self.app_label = app_label or app_name
        self.persona = persona if isinstance(persona, dict) else {}
        self.contacts: list[dict[str, Any]] = []

        self.setWindowTitle(f"聊天记录 - {device_ip} / {self.app_label}")
        self.resize(920, 640)
        self.setStyleSheet(
            """
            QDialog {
                background: #f8fafc;
                color: #0f172a;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            }
            QLabel#Title {
                color: #0f172a;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#Subtitle {
                color: #64748b;
                font-size: 12px;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 6px;
                outline: none;
            }
            QListWidget::item {
                border-radius: 6px;
                padding: 9px 8px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: #ffedd5;
                color: #0f172a;
            }
            QScrollArea {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: #ffffff;
            }
            QFrame#IncomingBubble {
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            QFrame#OutgoingBubble {
                background: #f97316;
                border: 1px solid #ea580c;
                border-radius: 8px;
            }
            QLabel#IncomingText {
                color: #0f172a;
                font-size: 13px;
                line-height: 1.4;
            }
            QLabel#OutgoingText {
                color: #ffffff;
                font-size: 13px;
                line-height: 1.4;
            }
            QLabel#MessageTime {
                color: #94a3b8;
                font-size: 11px;
            }
            QPushButton {
                min-height: 30px;
                padding: 5px 12px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                background: #ffffff;
                color: #334155;
            }
            QPushButton:hover {
                background: #f1f5f9;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("聊天记录")
        title.setObjectName("Title")
        subtitle = QLabel(f"{device_ip} / {self.app_label} / {self._persona_label()}")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.reload)
        header.addWidget(refresh)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        self.contact_list = QListWidget()
        self.contact_list.currentRowChanged.connect(self._show_contact)
        splitter.addWidget(self.contact_list)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(14, 14, 14, 14)
        self.messages_layout.setSpacing(10)
        self.scroll.setWidget(self.messages_widget)
        splitter.addWidget(self.scroll)
        splitter.setSizes([260, 660])

        self.reload()

    def _persona_label(self) -> str:
        name = str(self.persona.get("name", "") or "").strip()
        persona_id = str(self.persona.get("requestPersona", "") or "").strip()
        if name and persona_id:
            return f"{name} / {persona_id}"
        return name or persona_id or "默认人设"

    def reload(self) -> None:
        self.contacts = self.store.list_contacts(
            device_ip=self.device_ip,
            app_name=self.app_name,
            persona=self.persona,
        )
        current_contact = self._current_contact_name()
        self.contact_list.blockSignals(True)
        self.contact_list.clear()
        selected_row = 0
        for index, contact in enumerate(self.contacts):
            contact_name = str(contact.get("contact_name", "") or "")
            item = QListWidgetItem(self._contact_text(contact))
            item.setData(Qt.ItemDataRole.UserRole, contact_name)
            self.contact_list.addItem(item)
            if current_contact and contact_name == current_contact:
                selected_row = index
        self.contact_list.blockSignals(False)
        if self.contacts:
            self.contact_list.setCurrentRow(selected_row)
            self._show_contact(selected_row)
        else:
            self._show_empty("当前设备、App、人设下暂无聊天记录")

    def _current_contact_name(self) -> str:
        item = self.contact_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    @staticmethod
    def _contact_text(contact: dict[str, Any]) -> str:
        name = str(contact.get("contact_name", "") or "")
        last_at = str(contact.get("last_message_at", "") or "")
        content = str(contact.get("last_message", "") or "")
        direction = str(contact.get("last_direction", "") or "")
        prefix = "我：" if direction == "outgoing" else ""
        if len(content) > 24:
            content = content[:24] + "..."
        return f"{name}\n{last_at}  {prefix}{content}"

    def _clear_messages(self) -> None:
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _show_empty(self, text: str) -> None:
        self._clear_messages()
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #64748b; padding: 80px 12px;")
        self.messages_layout.addWidget(label, 1)

    def _show_contact(self, row: int) -> None:
        if row < 0 or row >= len(self.contacts):
            if not self.contacts:
                self._show_empty("当前设备、App、人设下暂无聊天记录")
            return
        contact_name = str(self.contacts[row].get("contact_name", "") or "")
        messages = self.store.get_messages(
            device_ip=self.device_ip,
            app_name=self.app_name,
            contact_name=contact_name,
            persona=self.persona,
        )
        self._clear_messages()
        if not messages:
            self._show_empty("该联系人暂无聊天记录")
            return
        for message in messages:
            self.messages_layout.addWidget(self._message_row(message))
        self.messages_layout.addStretch(1)

    def _message_row(self, message: dict[str, Any]) -> QWidget:
        is_outgoing = str(message.get("direction", "")) == "outgoing"
        bubble = self._bubble(message, is_outgoing)
        row = QWidget()
        # 让每一行占满气泡计算出的实际高度，避免下一条消息覆盖当前气泡。
        row.setMinimumHeight(max(1, bubble.sizeHint().height()))
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(
            bubble,
            0,
            Qt.AlignmentFlag.AlignRight if is_outgoing else Qt.AlignmentFlag.AlignLeft,
        )
        return row

    def _bubble(self, message: dict[str, Any], is_outgoing: bool) -> QFrame:
        bubble = QFrame()
        bubble.setObjectName("OutgoingBubble" if is_outgoing else "IncomingBubble")
        # QLabel 只有 maximumWidth 时，Qt 可能按未换行的 sizeHint 计算高度，
        # 长消息会因此覆盖下一条气泡。固定内容列宽，确保高度按换行后的尺寸计算。
        bubble_width = 464
        bubble.setFixedWidth(bubble_width)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(bubble)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(6)
        text = QLabel(str(message.get("content", "") or ""))
        text.setObjectName("OutgoingText" if is_outgoing else "IncomingText")
        text.setWordWrap(True)
        text.setFixedWidth(440)
        text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        time_label = QLabel(str(message.get("created_at", "") or ""))
        time_label.setObjectName("MessageTime")
        time_label.setAlignment(Qt.AlignmentFlag.AlignRight if is_outgoing else Qt.AlignmentFlag.AlignLeft)
        time_label.setFixedWidth(440)
        time_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(text)
        layout.addWidget(time_label)
        return bubble
