import { Component, OnInit, ElementRef, ViewChild } from '@angular/core';
import { SocketService } from '../socket.service';
import { SignalingService } from '../signaling.service';

@Component({
  selector: 'app-chat-home',
  templateUrl: './chat-home.component.html',
  styleUrls: ['./chat-home.component.scss']
})
export class ChatHomeComponent implements OnInit {
  displayName: string = '';
  @ViewChild('remoteVideo') remoteVideo: ElementRef;
  message: string = '';
  selectedRecipient; // Default: send to all users
  users: string[] = [];
  
  constructor(public socketService: SocketService,
              private signalingService: SignalingService) {}


  ngOnInit(): void {
    this.socketService.getUsers().subscribe(users => {
      this.users = users;
    });
    this.signalingService
      .getMessages()
      .subscribe((payload) => this._handleMessage(payload));
  }

  public async makeCall(): Promise<void> {
    await this.socketService.makeCall(this.remoteVideo);
  }
  private async _handleMessage(data): Promise<void> {
    switch (data.type) {
      case 'offer':
        await this.socketService.handleOffer(data.offer, this.remoteVideo);
        break;

      case 'answer':
        await this.socketService.handleAnswer(data.answer);
        break;

      case 'candidate':
        this.socketService.handleCandidate(data.candidate);
        break;

      default:
        break;
    }
  }
  changeChat(selectedUser){
    this.selectedRecipient = selectedUser;
  }

  test(){
    this.socketService.test();
  }
  join(){
    this.socketService.joinUser(this.displayName);
  }

  sendMessage() {
    if (this.message.trim() !== '') {
      this.socketService.sendMessage(this.message, this.displayName, this.selectedRecipient);
      this.message = '';
    }
  }
}
