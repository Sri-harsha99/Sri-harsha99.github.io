import { Component, OnInit } from '@angular/core';
import { SocketService } from '../socket.service';

@Component({
  selector: 'app-chat-home',
  templateUrl: './chat-home.component.html',
  styleUrls: ['./chat-home.component.scss']
})
export class ChatHomeComponent implements OnInit {
  // displayName: string = '';
  // message: string = '';
  // selectedRecipient; // Default: send to all users
  // users: string[] = [];
  
  // constructor(public socketService: SocketService) {}


  ngOnInit(): void {
    // this.socketService.getUsers().subscribe(users => {
    //   this.users = users;
    // });
  }
  // changeChat(selectedUser){
  //   this.selectedRecipient = selectedUser;
  // }

  // test(){
  //   this.socketService.test();
  // }
  // join(){
  //   this.socketService.joinUser(this.displayName);
  // }

  // sendMessage() {
  //   if (this.message.trim() !== '') {
  //     this.socketService.sendMessage(this.message, this.displayName, this.selectedRecipient);
  //     this.message = '';
  //   }
  // }
}
